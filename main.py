import logging
import asyncio
import io
import os
import time
from functools import wraps
import json
import docx
import google.generativeai as genai
from datetime import datetime

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardRemove
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
import fitz  # PyMuPDF

# ⚡️ асинхронный клиент Upstash Redis
try:
    from upstash_redis.asyncio import Redis
except Exception:
    # fallback на синхронный (не рекомендуется, но чтобы файл не падал при импорте)
    from upstash_redis import Redis  # type: ignore

# --- Настройка ---
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
ALLOWED_USER_IDS_STR = os.environ.get('ALLOWED_USER_IDS')
ALLOWED_USER_IDS = [int(user_id.strip()) for user_id in ALLOWED_USER_IDS_STR.split(',')] if ALLOWED_USER_IDS_STR else []

TELEGRAM_MAX_MESSAGE_LENGTH = 4096
DOCUMENT_ANALYSIS_MODELS = ['gemini-1.5-pro', 'gemini-2.5-pro']
IMAGE_GEN_MODELS = ['gemini-2.5-flash-image-preview']
HISTORY_LIMIT = 10
DEFAULT_CHAT_NAME = "default"

# --- Подключение к Upstash Redis (async) ---
redis_client = None
try:
    redis_client = Redis.from_url(
        url=os.environ.get('UPSTASH_REDIS_URL'),
        token=os.environ.get('UPSTASH_REDIS_TOKEN')
    )
except Exception as e:
    logging.error(f"Не удалось инициализировать Redis: {e}")
    redis_client = None

# --- Настройка логирования и Gemini API ---
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# --- Утилиты Redis (асинхронные врапперы) ---
async def r_get(key, default=None):
    try:
        if redis_client:
            val = await redis_client.get(key)
            return val if val is not None else default
    except Exception as e:
        logger.error(f"Redis GET error: {e}")
    return default

async def r_set(key, value, ex=None):
    try:
        if redis_client:
            await redis_client.set(key, value, ex=ex)
    except Exception as e:
        logger.error(f"Redis SET error: {e}")

async def r_sadd(key, value):
    try:
        if redis_client:
            await redis_client.sadd(key, value)
    except Exception as e:
        logger.error(f"Redis SADD error: {e}")

async def r_smembers(key) -> set:
    try:
        if redis_client:
            res = await redis_client.smembers(key)
            return set(res or [])
    except Exception as e:
        logger.error(f"Redis SMEMBERS error: {e}")
    return set()

async def r_delete(key):
    try:
        if redis_client:
            await redis_client.delete(key)
    except Exception as e:
        logger.error(f"Redis DEL error: {e}")

async def r_incrby(key, amount):
    try:
        if redis_client:
            await redis_client.incrby(key, amount)
    except Exception as e:
        logger.error(f"Redis INCRBY error: {e}")

async def r_expire(key, seconds):
    try:
        if redis_client:
            await redis_client.expire(key, seconds)
    except Exception as e:
        logger.error(f"Redis EXPIRE error: {e}")

# --- Декоратор для проверки авторизации ---
def restricted(func):
    @wraps(func)
    async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        user_id = update.effective_user.id
        if user_id not in ALLOWED_USER_IDS:
            logger.warning(f"Неавторизованный доступ отклонен для пользователя с ID: {user_id}")
            if update.message:
                await update.message.reply_text("⛔️ У вас нет доступа к этому боту.")
            elif update.callback_query:
                await update.callback_query.answer("⛔️ У вас нет доступа.", show_alert=True)
            return
        return await func(update, context, *args, **kwargs)
    return wrapped

# --- Вспомогательные функции ---
async def update_usage_stats(user_id: int, usage_metadata):
    # usage_metadata у стримов отсутствует — не ломаемся
    if not redis_client or not hasattr(usage_metadata, 'total_token_count'):
        return
    try:
        total_tokens = usage_metadata.total_token_count
        today = datetime.utcnow().strftime('%Y-%m-%d')
        daily_key = f"usage:{user_id}:daily:{today}"
        await r_incrby(daily_key, total_tokens)
        await r_expire(daily_key, 86400 * 2)
        this_month = datetime.utcnow().strftime('%Y-%m')
        monthly_key = f"usage:{user_id}:monthly:{this_month}"
        await r_incrby(monthly_key, total_tokens)
        await r_expire(monthly_key, 86400 * 32)
    except Exception as e:
        logger.error(f"Ошибка обновления статистики использования: {e}")

async def send_long_message(message, text: str):
    if not text or not text.strip():
        return
    # чтобы не возиться с экранированием Markdown, используем HTML
    for i in range(0, len(text), TELEGRAM_MAX_MESSAGE_LENGTH):
        await message.reply_text(text[i:i + TELEGRAM_MAX_MESSAGE_LENGTH], parse_mode=ParseMode.HTML)
        await asyncio.sleep(0.2)

def to_gemini_inline_image(pil_image: Image.Image) -> dict:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return {"inline_data": {"mime_type": "image/png", "data": buf.getvalue()}}

def bytes_to_inline_image(data: bytes, mime: str) -> dict:
    return {"inline_data": {"mime_type": mime, "data": data}}

# --- История/профиль пользователя ---
async def get_active_chat_name(user_id: int) -> str:
    return await r_get(f"active_chat:{user_id}", DEFAULT_CHAT_NAME)

async def get_history(user_id: int) -> list:
    active_chat = await get_active_chat_name(user_id)
    try:
        history_data = await r_get(f"history:{user_id}:{active_chat}")
        return json.loads(history_data) if history_data else []
    except Exception:
        return []

async def update_history(user_id: int, user_message_text: str, model_response_text: str):
    active_chat = await get_active_chat_name(user_id)
    history = await get_history(user_id)
    history.append({'role': 'user', 'parts': [{'text': user_message_text}]})
    history.append({'role': 'model', 'parts': [{'text': model_response_text}]})
    if len(history) > HISTORY_LIMIT:
        history = history[-HISTORY_LIMIT:]
    await r_set(f"history:{user_id}:{active_chat}", json.dumps(history), ex=86400 * 7)

async def get_user_model(user_id: int) -> str:
    return await r_get(f"user:{user_id}:model", "gemini-1.5-flash")

async def get_user_persona(user_id: int) -> str | None:
    return await r_get(f"persona:{user_id}")

# --- Главное меню и подменю ---
async def get_main_menu_text_and_keyboard(user_id: int):
    model_name = await get_user_model(user_id)
    active_chat = await get_active_chat_name(user_id)
    text = (
        f"🤖 <b>Главное меню</b>\n\n"
        f"Текущая модель: <code>{model_name}</code>\n"
        f"Текущий чат: <code>{active_chat}</code>\n\n"
        f"Выберите действие:"
    )
    keyboard = [
        [
            InlineKeyboardButton("🤖 Выбрать модель", callback_data="menu:model"),
            InlineKeyboardButton("👤 Персона", callback_data="menu:persona")
        ],
        [
            InlineKeyboardButton("💬 Управление чатами", callback_data="menu:open_chats_submenu")
        ],
        [
            InlineKeyboardButton("🗑️ Очистить текущий чат", callback_data="menu:clear"),
            InlineKeyboardButton("📈 Статистика", callback_data="menu:usage")
        ],
        [
            InlineKeyboardButton("❓ Что умеет бот?", callback_data="menu:help")
        ]
    ]
    return text, InlineKeyboardMarkup(keyboard)

async def get_chats_submenu_text_and_keyboard():
    text = "🗂️ <b>Управление чатами</b>"
    keyboard = [
        [InlineKeyboardButton("📖 Сохраненные чаты", callback_data="chats:list")],
        [InlineKeyboardButton("📥 Сохранить текущий чат", callback_data="chats:save")],
        [InlineKeyboardButton("➕ Новый чат", callback_data="chats:new")],
        [InlineKeyboardButton("⬅️ Назад в меню", callback_data="menu:main")]
    ]
    return text, InlineKeyboardMarkup(keyboard)

# --- Обработчики команд/кнопок ---
@restricted
async def main_menu_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    menu_text, reply_markup = await get_main_menu_text_and_keyboard(user_id)

    # если команда пришла как текст — удалим её, чтобы оставить чистое меню
    if update.message:
        try:
            await update.message.delete()
        except Exception:
            pass

    target_message = update.callback_query.message if update.callback_query else None

    try:
        if target_message:
            await target_message.edit_text(menu_text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
        else:
            await context.bot.send_message(chat_id=user_id, text=menu_text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
    except Exception as e:
        if "Message is not modified" not in str(e):
            await context.bot.send_message(chat_id=user_id, text=menu_text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)

async def clear_history_logic(update: Update):
    user_id = update.effective_user.id
    active_chat = await get_active_chat_name(user_id)
    await r_delete(f"history:{user_id}:{active_chat}")
    return f"Память текущего чата (<code>{active_chat}</code>) очищена."

@restricted
async def clear_history_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    response_text = await clear_history_logic(update)
    await update.message.reply_text(response_text, parse_mode=ParseMode.HTML)

@restricted
async def usage_command(update: Update, context: ContextTypes.DEFAULT_TYPE, from_callback: bool = False):
    user_id = update.effective_user.id
    if not redis_client:
        target = update.callback_query if from_callback else update.message
        await target.reply_text("Хранилище не подключено, статистика недоступна.")  # type: ignore
        return
    today = datetime.utcnow().strftime('%Y-%m-%d')
    this_month = datetime.utcnow().strftime('%Y-%m')
    daily_tokens = await r_get(f"usage:{user_id}:daily:{today}", 0)
    monthly_tokens = await r_get(f"usage:{user_id}:monthly:{this_month}", 0)
    try:
        daily_tokens = int(daily_tokens)
    except Exception:
        daily_tokens = 0
    try:
        monthly_tokens = int(monthly_tokens)
    except Exception:
        monthly_tokens = 0

    text = (
        f"📊 <b>Статистика использования токенов:</b>\n\n"
        f"Сегодня ({today}):\n<code>{daily_tokens:,}</code> токенов\n\n"
        f"В этом месяце ({this_month}):\n<code>{monthly_tokens:,}</code> токенов"
    )
    if from_callback:
        keyboard = [[InlineKeyboardButton("⬅️ Назад в меню", callback_data='menu:main')]]
        await update.callback_query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.HTML)
    else:
        await update.message.reply_text(text, parse_mode=ParseMode.HTML)

@restricted
async def persona_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    persona_text = " ".join(context.args) if context.args else None
    if not redis_client:
        await update.message.reply_text("Хранилище не подключено, не могу сохранить персону.")
        return
    if persona_text:
        await r_set(f"persona:{user_id}", persona_text)
        await update.message.reply_text(f"✅ Новая персона установлена:\n\n<i>{persona_text}</i>", parse_mode=ParseMode.HTML)
    else:
        await r_delete(f"persona:{user_id}")
        await update.message.reply_text("🗑️ Персона сброшена до стандартной.")

@restricted
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE, from_callback: bool = False):
    help_text = (
        "Я ваш персональный ассистент, подключенный к Google Gemini.\n\n"
        "💬 <b>Просто общайтесь со мной</b>\n"
        "Пишите любой вопрос или задачу, я помогаю и помню контекст.\n\n"
        "🤖 <b>Выбор модели</b> через меню: <code>Pro</code> — сложные задачи, <code>Flash</code> — быстро, "
        "<code>Image Preview</code> — генерация/анализ изображений.\n\n"
        "👤 <b>Персона</b> (/persona): задайте стиль ответа.\n\n"
        "🗂️ <b>Управление диалогами</b>:\n"
        "• /new_chat — новый диалог\n"
        "• /save_chat &lt;имя&gt; — сохранить текущий\n"
        "• /load_chat &lt;имя&gt; — загрузить сохранённый\n"
        "• /chats — список чатов\n"
        "• /delete_chat &lt;имя&gt; — удалить чат\n"
        "• /clear — очистить историю активного диалога\n\n"
        "🖼️ <b>Изображения</b>: генерация и анализ фото (в зависимости от модели).\n\n"
        "📄 <b>Документы</b>: отправьте PDF/DOCX/TXT с подписью-заданием.\n\n"
        "📈 <b>Расходы</b> (/usage): статистика токенов за сегодня и месяц.\n"
    )
    if from_callback:
        keyboard = [[InlineKeyboardButton("⬅️ Назад в меню", callback_data='menu:main')]]
        await update.callback_query.edit_message_text(help_text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.HTML)
    else:
        await update.message.reply_text(help_text, parse_mode=ParseMode.HTML)

@restricted
async def model_selection_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("Gemini 2.5 Pro", callback_data='select_model:gemini-2.5-pro')],
        [InlineKeyboardButton("Gemini 1.5 Pro", callback_data='select_model:gemini-1.5-pro')],
        [InlineKeyboardButton("Gemini 2.5 Flash", callback_data='select_model:gemini-2.5-flash')],
        [InlineKeyboardButton("Gemini 1.5 Flash", callback_data='select_model:gemini-1.5-flash')],
        [InlineKeyboardButton("Image Preview", callback_data='select_model:gemini-2.5-flash-image-preview')],
        [InlineKeyboardButton("⬅️ Назад в меню", callback_data='menu:main')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.edit_message_text('Выберите модель:', reply_markup=reply_markup)

@restricted
async def new_chat_command(update: Update, context: ContextTypes.DEFAULT_TYPE, from_callback: bool = False):
    user_id = update.effective_user.id
    if not redis_client:
        return
    await r_set(f"active_chat:{user_id}", DEFAULT_CHAT_NAME)
    await r_delete(f"history:{user_id}:{DEFAULT_CHAT_NAME}")
    response_text = f"Начат новый диалог (<code>{DEFAULT_CHAT_NAME}</code>)."
    target_message = update.callback_query.message if from_callback else update.message
    await target_message.reply_text(response_text, parse_mode=ParseMode.HTML)

@restricted
async def save_chat_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not redis_client:
        return
    chat_name = "_".join(context.args).strip().replace(" ", "_")
    if not chat_name or chat_name == DEFAULT_CHAT_NAME:
        await update.message.reply_text("Пожалуйста, укажите имя для сохранения. Например: <code>/save_chat мой_проект</code>.", parse_mode=ParseMode.HTML)
        return
    active_chat = await get_active_chat_name(user_id)
    current_history_json = await r_get(f"history:{user_id}:{active_chat}")
    if not current_history_json:
        await update.message.reply_text("Текущий диалог пуст, нечего сохранять.")
        return
    await r_set(f"history:{user_id}:{chat_name}", current_history_json, ex=86400 * 7)
    await r_sadd(f"chats:{user_id}", chat_name)
    await r_set(f"active_chat:{user_id}", chat_name)
    await update.message.reply_text(f"Текущий диалог сохранен как <code>{chat_name}</code> и сделан активным.", parse_mode=ParseMode.HTML)

@restricted
async def load_chat_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not redis_client:
        return
    chat_name = "_".join(context.args).strip().replace(" ", "_")
    if not chat_name:
        await update.message.reply_text("Пожалуйста, укажите имя чата для загрузки. Например: <code>/load_chat мой_проект</code>.", parse_mode=ParseMode.HTML)
        return
    all_chats = await r_smembers(f"chats:{user_id}")
    if chat_name != DEFAULT_CHAT_NAME and chat_name not in all_chats:
        await update.message.reply_text(f"Чата с именем <code>{chat_name}</code> не найдено.", parse_mode=ParseMode.HTML)
        return
    await r_set(f"active_chat:{user_id}", chat_name)
    await update.message.reply_text(f"Чат <code>{chat_name}</code> загружен и сделан активным.", parse_mode=ParseMode.HTML)

@restricted
async def list_chats_command(update: Update, context: ContextTypes.DEFAULT_TYPE, from_callback: bool = False):
    user_id = update.effective_user.id
    if not redis_client:
        return
    active_chat = await get_active_chat_name(user_id)
    all_chats = await r_smembers(f"chats:{user_id}")
    message = f"<b>Ваши диалоги:</b>\n\n"
    if active_chat == DEFAULT_CHAT_NAME:
        message += f"➡️ <code>{DEFAULT_CHAT_NAME}</code> (активный)\n"
    else:
        message += f"▫️ <code>{DEFAULT_CHAT_NAME}</code> (<code>/new_chat</code>)\n"
    for chat in sorted(list(all_chats)):
        if chat == active_chat:
            message += f"➡️ <code>{chat}</code> (активный)\n"
        else:
            message += f"▫️ <code>{chat}</code> (<code>/load_chat {chat}</code>)\n"

    keyboard = [[InlineKeyboardButton("⬅️ Назад в меню", callback_data='menu:main')]]
    if from_callback:
        await update.callback_query.edit_message_text(message, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.HTML)
    else:
        await update.message.reply_text(message, parse_mode=ParseMode.HTML)

@restricted
async def delete_chat_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not redis_client:
        return
    chat_name = "_".join(context.args).strip().replace(" ", "_")
    if not chat_name or chat_name == DEFAULT_CHAT_NAME:
        await update.message.reply_text("Нельзя удалить чат по умолчанию. Укажите имя, например: <code>/delete_chat мой_проект</code>.", parse_mode=ParseMode.HTML)
        return
    all_chats = await r_smembers(f"chats:{user_id}")
    if chat_name not in all_chats:
        await update.message.reply_text(f"Чата с именем <code>{chat_name}</code> не найдено.", parse_mode=ParseMode.HTML)
        return
    await r_delete(f"history:{user_id}:{chat_name}")
    try:
        # в upstash_redis.asyncio нет srem удобного? есть, но через eval проще обойти
        await redis_client.srem(f"chats:{user_id}", chat_name)  # type: ignore
    except Exception:
        pass
    active_chat = await get_active_chat_name(user_id)
    if active_chat == chat_name:
        await r_set(f"active_chat:{user_id}", DEFAULT_CHAT_NAME)
        await update.message.reply_text(f"Чат <code>{chat_name}</code> удален. Вы переключены на чат по умолчанию.", parse_mode=ParseMode.HTML)
    else:
        await update.message.reply_text(f"Чат <code>{chat_name}</code> удален.", parse_mode=ParseMode.HTML)

@restricted
async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    command, *payload = query.data.split(':', 1)
    payload = payload[0] if payload else None

    if command == "menu":
        if payload == "model":
            await model_selection_menu(update, context)
        elif payload == "persona":
            await query.message.reply_text("Отправьте команду:\n<code>/persona &lt;текст&gt;</code> для установки,\n<code>/persona</code> без текста для сброса.", parse_mode=ParseMode.HTML)
        elif payload == "open_chats_submenu":
            submenu_text, reply_markup = await get_chats_submenu_text_and_keyboard()
            await query.edit_message_text(submenu_text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
        elif payload == "clear":
            response_text = await clear_history_logic(update)
            await query.message.reply_text(response_text, parse_mode=ParseMode.HTML)
            await main_menu_command(update, context)  # ✅ фикс: раньше здесь был несуществующий menu_command
        elif payload == "usage":
            await usage_command(update, context, from_callback=True)
        elif payload == "help":
            await help_command(update, context, from_callback=True)
        elif payload == "main":
            await main_menu_command(update, context)

    elif command == "chats":
        if payload == "list":
            await list_chats_command(update, context, from_callback=True)
        elif payload == "save":
            await query.message.reply_text("Чтобы сохранить текущий чат, отправьте команду:\n<code>/save_chat &lt;имя_чата&gt;</code>\nПробелы будут заменены на подчеркивания.", parse_mode=ParseMode.HTML)
        elif payload == "new":
            await new_chat_command(update, context, from_callback=True)
            await main_menu_command(update, context)

    elif command == "select_model":
        user_id = query.from_user.id
        if redis_client:
            await r_set(f"user:{user_id}:model", payload)
        menu_text, reply_markup = await get_main_menu_text_and_keyboard(user_id)
        try:
            await query.edit_message_text(
                f"✅ Модель изменена на <code>{payload}</code>.\n\n" + menu_text,
                reply_markup=reply_markup,
                parse_mode=ParseMode.HTML
            )
        except Exception:
            pass

# --- Обработка ответов Gemini ---
async def handle_gemini_response(update: Update, response):
    """Обрабатывает НЕ-стриминговые ответы (фото, документы, генерация изображений)."""
    try:
        if hasattr(response, 'usage_metadata'):
            await update_usage_stats(update.effective_user.id, response.usage_metadata)

        if not getattr(response, "candidates", None):
            reason = getattr(getattr(response, "prompt_feedback", None), 'block_reason_message', 'Причина не указана.')
            await update.message.reply_text(f"⚠️ Запрос был заблокирован.\nПричина: {reason}")
            return

        candidate = response.candidates[0]
        finish = getattr(candidate, "finish_reason", None)
        if finish and getattr(finish, "name", "STOP") != "STOP":
            await update.message.reply_text(f"⚠️ Контент не может быть сгенерирован. Причина: <code>{finish.name}</code>", parse_mode=ParseMode.HTML)
            return

        parts = getattr(candidate, "content", None).parts if getattr(candidate, "content", None) else []
        full_text = ""
        image_sent = False
        for part in parts:
            # текст
            if hasattr(part, 'text') and part.text:
                full_text += part.text
            # картинка в inline_data
            elif hasattr(part, 'inline_data') and part.inline_data and getattr(part.inline_data, "mime_type", "").startswith('image/'):
                try:
                    data = part.inline_data.data if hasattr(part.inline_data, "data") else None
                    if data:
                        await update.message.reply_photo(photo=io.BytesIO(data))
                        image_sent = True
                except Exception as e:
                    logger.error(f"Ошибка отправки изображения: {e}")

        if full_text.strip() and not image_sent:
            await send_long_message(update.message, full_text)

    except Exception as e:
        logger.error(f"Критическая ошибка при обработке ответа от Gemini: {e}")
        await update.message.reply_text(f"Произошла критическая ошибка при обработке ответа: {e}")

async def handle_gemini_response_stream(update: Update, response_stream, user_message_text: str):
    """Обрабатывает потоковый ответ, обновляя плейсхолдер и отправляя финал."""
    placeholder_message = None
    full_response_text = ""
    last_update_time = 0
    update_interval = 0.8
    try:
        placeholder_message = await update.message.reply_text("...")
        last_update_time = time.time()

        async for chunk in response_stream:
            if hasattr(chunk, 'text') and chunk.text:
                full_response_text += chunk.text
                current_time = time.time()
                if current_time - last_update_time > update_interval:
                    try:
                        # промежуточные апдейты без parse_mode — чтобы избежать лишних ошибок форматирования
                        if len(full_response_text) < TELEGRAM_MAX_MESSAGE_LENGTH - 10:
                            await placeholder_message.edit_text(full_response_text + " ✍️")
                            last_update_time = current_time
                    except Exception:
                        pass

        if placeholder_message:
            try:
                await placeholder_message.delete()
            except Exception:
                pass

        if not full_response_text.strip():
            await update.message.reply_text("Модель завершила работу, но не сгенерировала ответ. Попробуйте переформулировать ваш запрос.")
            return

        await send_long_message(update.message, full_response_text)
        await update_history(update.effective_user.id, user_message_text, full_response_text)

        # response_stream часто не содержит usage_metadata — пропускаем без ошибки

    except Exception as e:
        logger.error(f"Критическая ошибка при обработке стриминг-ответа от Gemini: {e}")
        if placeholder_message:
            try:
                await placeholder_message.delete()
            except Exception:
                pass
        await update.message.reply_text(f"Произошла ошибка при генерации ответа: {e}")

# --- Хэндлеры сообщений ---
@restricted
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_message = update.message.text
    model_name = await get_user_model(user_id)
    persona = await get_user_persona(user_id)

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    try:
        model = genai.GenerativeModel(model_name, system_instruction=persona)
        if model_name in IMAGE_GEN_MODELS:
            image_prompt = f"Generate a high-quality, photorealistic image of: {user_message}"
            response = await model.generate_content_async(image_prompt)
            await handle_gemini_response(update, response)
            await update_history(user_id, user_message, "[Запрос на генерацию изображения]")
        else:
            history = await get_history(user_id)
            chat = model.start_chat(history=history)
            response_stream = await chat.send_message_async(user_message, stream=True)
            await handle_gemini_response_stream(update, response_stream, user_message)
    except Exception as e:
        logger.error(f"Ошибка при обработке текстового сообщения: {e}")
        await update.message.reply_text(f'К сожалению, произошла ошибка: {e}')

@restricted
async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    model_name = await get_user_model(user_id)
    persona = await get_user_persona(user_id)
    if model_name not in IMAGE_GEN_MODELS:
        await update.message.reply_text("Чтобы работать с фото, выберите модель \"Image Preview\" через /menu.")
        return

    photo_file = await update.message.photo[-1].get_file()
    caption = update.message.caption or "Опиши это изображение"

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_PHOTO)

    try:
        # скачиваем байты без PIL, и отправляем их как inline_data
        photo_bytes = io.BytesIO()
        await photo_file.download_to_memory(photo_bytes)
        photo_bytes.seek(0)
        inline_img = bytes_to_inline_image(photo_bytes.getvalue(), "image/jpeg")

        model_gemini = genai.GenerativeModel(model_name, system_instruction=persona)
        response = await model_gemini.generate_content_async([caption, inline_img])
        await handle_gemini_response(update, response)
    except Exception as e:
        logger.error(f"Ошибка при обработке фото: {e}")
        await update.message.reply_text(f'К сожалению, произошла ошибка при обработке фото: {e}')

@restricted
async def handle_document_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    model_name = await get_user_model(user_id)
    persona = await get_user_persona(user_id)
    if model_name not in DOCUMENT_ANALYSIS_MODELS:
        await update.message.reply_text(f"Для анализа документов, пожалуйста, выберите модель Pro.")
        return

    doc = update.message.document
    caption = update.message.caption or "Проанализируй этот документ и сделай краткую выжимку."
    await update.message.reply_text(f"Получил файл: {doc.file_name}.\nНачинаю обработку...")

    try:
        doc_file = await doc.get_file()
        file_bytes_io = io.BytesIO()
        await doc_file.download_to_memory(file_bytes_io)
        file_bytes_io.seek(0)

        content_parts = [caption]

        if doc.mime_type == 'application/pdf':
            pdf_document = fitz.open(stream=file_bytes_io.read(), filetype="pdf")
            page_limit = 25
            num_pages = min(len(pdf_document), page_limit)
            for page_num in range(num_pages):
                page = pdf_document.load_page(page_num)
                pix = page.get_pixmap()
                img_bytes = pix.tobytes("png")
                # без PIL: сразу inline image
                content_parts.append(bytes_to_inline_image(img_bytes, "image/png"))
            pdf_document.close()
            await update.message.reply_text(f"Отправляю первые {num_pages} страниц PDF в Gemini на анализ...")

        elif doc.mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            document = docx.Document(file_bytes_io)
            file_text_content = "\n".join([para.text for para in document.paragraphs])
            content_parts.append(file_text_content)

        elif doc.mime_type == 'text/plain':
            file_text_content = file_bytes_io.read().decode('utf-8', errors='ignore')
            content_parts.append(file_text_content)

        else:
            await update.message.reply_text(f"Извините, я пока не поддерживаю файлы типа {doc.mime_type}.")
            return

        model = genai.GenerativeModel(model_name, system_instruction=persona)
        response = await model.generate_content_async(content_parts)
        await handle_gemini_response(update, response)

    except Exception as e:
        logger.error(f"Ошибка при обработке документа: {e}")
        await update.message.reply_text(f'К сожалению, произошла ошибка при обработке документа: {e}')

# --- Регистрация и запуск бота ---
def main() -> None:
    logger.info("Создание и настройка приложения...")
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # команды
    application.add_handler(CommandHandler("start", main_menu_command))
    application.add_handler(CommandHandler("menu", main_menu_command))
    application.add_handler(CommandHandler("clear", clear_history_command))
    application.add_handler(CommandHandler("usage", usage_command))
    application.add_handler(CommandHandler("persona", persona_command))
    application.add_handler(CommandHandler("new_chat", new_chat_command))
    application.add_handler(CommandHandler("save_chat", save_chat_command))
    application.add_handler(CommandHandler("load_chat", load_chat_command))
    application.add_handler(CommandHandler("chats", list_chats_command))
    application.add_handler(CommandHandler("delete_chat", delete_chat_command))
    application.add_handler(CommandHandler("help", help_command))

    # колбэки кнопок
    application.add_handler(CallbackQueryHandler(button_callback))

    # сообщения
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo_message))
    supported_files_filter = filters.Document.PDF | filters.Document.DOCX | filters.Document.TXT
    application.add_handler(MessageHandler(supported_files_filter, handle_document_message))

    logger.info("Бот запущен и работает в режиме опроса...")
    application.run_polling()

if __name__ == "__main__":
    # Проверяем переменные окружения и подключение к Redis (пинг только если есть метод и клиент)
    ready = True
    missing = []
    if not TELEGRAM_BOT_TOKEN: missing.append("TELEGRAM_BOT_TOKEN")
    if not GEMINI_API_KEY: missing.append("GEMINI_API_KEY")
    if not ALLOWED_USER_IDS_STR: missing.append("ALLOWED_USER_IDS")
    if not redis_client: missing.append("UPSTASH_REDIS_URL/UPSTASH_REDIS_TOKEN")

    if missing:
        logger.critical(f"Не все переменные окружения настроены! Отсутствуют: {', '.join(missing)}")
        ready = False

    # Пингуем Redis асинхронно (если возможно)
    async def _ping():
        try:
            if hasattr(redis_client, "ping"):
                await redis_client.ping()  # type: ignore
                logger.info("Успешно подключено к Upstash Redis.")
        except Exception as e:
            logger.error(f"Не удалось подключиться к Redis: {e}")

    if ready:
        try:
            asyncio.run(_ping())
        except Exception:
            pass
        main()
