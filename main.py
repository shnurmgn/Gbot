import logging
import asyncio
import io
import os
import time
import json
import docx
import google.generativeai as genai
from datetime import datetime
from functools import wraps

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)
from telegram.constants import ChatAction, ParseMode
from PIL import Image
import fitz
from upstash_redis.asyncio import Redis  # ⚡ асинхронный клиент

# --- Настройка ---
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
ALLOWED_USER_IDS_STR = os.environ.get("ALLOWED_USER_IDS")
ALLOWED_USER_IDS = (
    [int(uid.strip()) for uid in ALLOWED_USER_IDS_STR.split(",")]
    if ALLOWED_USER_IDS_STR
    else []
)

TELEGRAM_MAX_MESSAGE_LENGTH = 4096
DOCUMENT_ANALYSIS_MODELS = ["gemini-1.5-pro", "gemini-2.5-pro"]
IMAGE_GEN_MODELS = ["gemini-2.5-flash-image-preview"]
HISTORY_LIMIT = 10
DEFAULT_CHAT_NAME = "default"

# --- Redis ---
redis_client: Redis | None = None
try:
    redis_client = Redis.from_url(
        url=os.environ.get("UPSTASH_REDIS_URL"),
        token=os.environ.get("UPSTASH_REDIS_TOKEN"),
    )
except Exception as e:
    logging.error(f"Redis init error: {e}")
    redis_client = None

# --- Логирование и Gemini ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# --- Декоратор для проверки доступа ---
def restricted(func):
    @wraps(func)
    async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE, *a, **kw):
        user_id = update.effective_user.id
        if user_id not in ALLOWED_USER_IDS:
            if update.message:
                await update.message.reply_text("⛔️ Нет доступа.")
            elif update.callback_query:
                await update.callback_query.answer("⛔️ Нет доступа.", show_alert=True)
            return
        return await func(update, context, *a, **kw)

    return wrapped


# --- Вспомогательные функции ---
async def redis_get(key, default=None):
    try:
        if redis_client:
            val = await redis_client.get(key)
            return val if val is not None else default
    except Exception:
        return default
    return default


async def redis_set(key, value, ex=None):
    try:
        if redis_client:
            await redis_client.set(key, value, ex=ex)
    except Exception:
        pass


async def redis_sadd(key, value):
    try:
        if redis_client:
            await redis_client.sadd(key, value)
    except Exception:
        pass


async def redis_smembers(key):
    try:
        if redis_client:
            return await redis_client.smembers(key)
    except Exception:
        return set()
    return set()


async def redis_delete(key):
    try:
        if redis_client:
            await redis_client.delete(key)
    except Exception:
        pass


async def update_usage_stats(user_id: int, usage_metadata):
    if not redis_client or not hasattr(usage_metadata, "total_token_count"):
        return
    try:
        total_tokens = usage_metadata.total_token_count
        today = datetime.utcnow().strftime("%Y-%m-%d")
        this_month = datetime.utcnow().strftime("%Y-%m")
        await redis_client.incrby(f"usage:{user_id}:daily:{today}", total_tokens)
        await redis_client.expire(f"usage:{user_id}:daily:{today}", 86400 * 2)
        await redis_client.incrby(f"usage:{user_id}:monthly:{this_month}", total_tokens)
        await redis_client.expire(f"usage:{user_id}:monthly:{this_month}", 86400 * 32)
    except Exception as e:
        logger.error(f"usage stats error: {e}")


async def send_long_message(message, text: str):
    if not text.strip():
        return
    for i in range(0, len(text), TELEGRAM_MAX_MESSAGE_LENGTH):
        await message.reply_text(
            text[i : i + TELEGRAM_MAX_MESSAGE_LENGTH], parse_mode=ParseMode.MARKDOWN_V2
        )
        await asyncio.sleep(0.3)


def pil_to_gemini_part(img: Image.Image):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return {"inline_data": {"mime_type": "image/png", "data": buf.getvalue()}}


# --- История ---
async def get_active_chat_name(user_id: int) -> str:
    return await redis_get(f"active_chat:{user_id}", DEFAULT_CHAT_NAME)


async def get_history(user_id: int) -> list:
    active_chat = await get_active_chat_name(user_id)
    data = await redis_get(f"history:{user_id}:{active_chat}")
    return json.loads(data) if data else []


async def update_history(user_id: int, user_msg: str, model_msg: str):
    active_chat = await get_active_chat_name(user_id)
    hist = await get_history(user_id)
    hist.append({"role": "user", "parts": [{"text": user_msg}]})
    hist.append({"role": "model", "parts": [{"text": model_msg}]})
    if len(hist) > HISTORY_LIMIT:
        hist = hist[-HISTORY_LIMIT:]
    await redis_set(f"history:{user_id}:{active_chat}", json.dumps(hist), ex=86400 * 7)


async def get_user_model(user_id: int) -> str:
    return await redis_get(f"user:{user_id}:model", "gemini-1.5-flash")


async def get_user_persona(user_id: int) -> str | None:
    return await redis_get(f"persona:{user_id}")


# --- Обработчики ---
@restricted
async def main_menu_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    model_name = await get_user_model(user_id)
    active_chat = await get_active_chat_name(user_id)
    text = f"🤖 Главное меню\n\nТекущая модель: `{model_name}`\nТекущий чат: `{active_chat}`"
    keyboard = [[InlineKeyboardButton("Выбрать модель", callback_data="menu:model")]]
    markup = InlineKeyboardMarkup(keyboard)

    if update.message:
        await update.message.reply_text(text, reply_markup=markup)
    elif update.callback_query:
        await update.callback_query.edit_message_text(text, reply_markup=markup)


@restricted
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    msg = update.message.text
    model_name = await get_user_model(user_id)
    persona = await get_user_persona(user_id)
    await context.bot.send_chat_action(update.effective_chat.id, ChatAction.TYPING)

    try:
        model = genai.GenerativeModel(model_name, system_instruction=persona)
        if model_name in IMAGE_GEN_MODELS:
            prompt = f"Generate photorealistic image: {msg}"
            resp = await model.generate_content_async(prompt)
            if hasattr(resp, "candidates") and resp.candidates:
                await update.message.reply_text("🖼 Картинка сгенерирована!")
            await update_history(user_id, msg, "[image gen]")
        else:
            history = await get_history(user_id)
            chat = model.start_chat(history=history)
            stream = await chat.send_message_async(msg, stream=True)
            full = ""
            async for chunk in stream:
                if hasattr(chunk, "text") and chunk.text:
                    full += chunk.text
            if full.strip():
                await send_long_message(update.message, full)
                await update_history(user_id, msg, full)
    except Exception as e:
        await update.message.reply_text(f"Ошибка: {e}")


# --- Главная точка ---
def main():
    logger.info("Bot starting...")
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", main_menu_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Polling...")
    app.run_polling()


if __name__ == "__main__":
    if not all([TELEGRAM_BOT_TOKEN, GEMINI_API_KEY, ALLOWED_USER_IDS_STR, redis_client]):
        logger.critical("Нет переменных окружения — бот не запущен.")
    else:
        main()
