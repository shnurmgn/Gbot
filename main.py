import logging
import asyncio
import io
import os
import time
from functools import wraps
import json
import docx
import google.generativeai as genai
import telegram
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)
from PIL import Image
import fitz
from upstash_redis import Redis

# --- Настройка (без изменений) ---
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
ALLOWED_USER_IDS_STR = os.environ.get('ALLOWED_USER_IDS')
ALLOWED_USER_IDS = [int(user_id.strip()) for user_id in ALLOWED_USER_IDS_STR.split(',')] if ALLOWED_USER_IDS_STR else []

TELEGRAM_MAX_MESSAGE_LENGTH = 4096
DOCUMENT_ANALYSIS_MODELS = ['gemini-1.5-pro', 'gemini-2.5-pro']
HISTORY_LIMIT = 10

# --- Подключение к Upstash Redis (без изменений) ---
redis_client = None
try:
    redis_client = Redis(
        url=os.environ.get('UPSTASH_REDIS_URL'),
        token=os.environ.get('UPSTASH_REDIS_TOKEN'),
        decode_responses=True
    )
    redis_client.ping()
    logging.info("Успешно подключено к Upstash Redis.")
except Exception as e:
    logging.error(f"Не удалось подключиться к Redis: {e}")

# --- Настройка логирования и Gemini API (без изменений) ---
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# --- Декоратор и Вспомогательные функции (с изменениями) ---

def restricted(func):
    @wraps(func)
    async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        user_id = update.effective_user.id
        if user_id not in ALLOWED_USER_IDS:
            if update.message: await update.message.reply_text("⛔️ У вас нет доступа к этому боту.")
            return
        return await func(update, context, *args, **kwargs)
    return wrapped

async def send_long_message(update: Update, text: str):
    if not text.strip(): return
    if len(text) <= TELEGRAM_MAX_MESSAGE_LENGTH:
        await update.message.reply_text(text)
    else:
        for i in range(0, len(text), TELEGRAM_MAX_MESSAGE_LENGTH):
            await update.message.reply_text(text[i:i + TELEGRAM_MAX_MESSAGE_LENGTH])
            await asyncio.sleep(0.5)

async def handle_gemini_response(update: Update, response):
    # Эта функция теперь используется только для НЕ-стриминговых ответов (фото, документы)
    try:
        full_text = ""
        for part in response.parts:
            if hasattr(part, 'text') and part.text:
                full_text += part.text
            elif hasattr(part, 'inline_data') and part.inline_data.mime_type.startswith('image/'):
                await update.message.reply_photo(photo=io.BytesIO(part.inline_data.data))
        
        if full_text:
            await send_long_message(update, full_text)

    except Exception as e:
        logger.error(f"Критическая ошибка при обработке ответа от Gemini: {e}")
        await update.message.reply_text(f"Произошла критическая ошибка при обработке ответа: {e}")

def get_history(user_id: int) -> list:
    if not redis_client: return []
    try:
        history_data = redis_client.get(f"history:{user_id}")
        return json.loads(history_data) if history_data else []
    except Exception: return []

def update_history(user_id: int, chat_history: list):
    if not redis_client: return
    # Конвертируем специальный объект chat.history в простой JSON
    history_to_save = [{'role': p.role, 'parts': [part.text for part in p.parts]} for p in chat_history]
    if len(history_to_save) > HISTORY_LIMIT:
        history_to_save = history_to_save[-HISTORY_LIMIT:]
    redis_client.set(f"history:{user_id}", json.dumps(history_to_save), ex=86400)

def get_user_model(user_id: int) -> str:
    default_model = 'gemini-1.5-flash'
    if not redis_client: return default_model
    try:
        stored_model = redis_client.get(f"user:{user_id}:model")
        return stored_model if stored_model else default_model
    except Exception: return default_model

# --- Функции-обработчики ---
@restricted
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    model_name = get_user_model(user.id)
    await update.message.reply_html(rf"Привет, {user.mention_html()}!")
    await update.message.reply_text(f"Я бот, подключенный к Gemini.\nТекущая модель: {model_name}.\n\nЧтобы начать новый диалог и очистить мою память, используйте команду /clear.")

@restricted
async def clear_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if redis_client: redis_client.delete(f"history:{user_id}")
    await update.message.reply_text("Память очищена.")

@restricted
async def model_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("Gemini 2.5 Pro (Документы/Текст)", callback_data='gemini-2.5-pro')],
        [InlineKeyboardButton("Gemini 1.5 Pro (Документы/Текст)", callback_data='gemini-1.5-pro')],
        [InlineKeyboardButton("Gemini 2.5 Flash (Текст)", callback_data='gemini-2.5-flash')],
        [InlineKeyboardButton("Gemini 1.5 Flash (Текст)", callback_data='gemini-1.5-flash')],
        [InlineKeyboardButton("Nano Banana (Изображения)", callback_data='gemini-2.5-flash-image-preview')],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text('Выберите модель:', reply_markup=reply_markup)

@restricted
async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    await query.answer()
    selected_model = query.data
    if redis_client: redis_client.set(f"user:{user_id}:model", selected_model)
    await query.edit_message_text(text=f"Модель изменена на: {selected_model}. Я запомню ваш выбор.")

@restricted
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_message = update.message.text
    model_name = get_user_model(user_id)
    
    placeholder_message = None
    try:
        history = get_history(user_id)
        model = genai.GenerativeModel(model_name)
        chat = model.start_chat(history=history)

        # Отправляем начальное сообщение-заготовку
        placeholder_message = await update.message.reply_text("...")
        full_response_text = ""
        last_update_time = 0
        update_interval = 0.8  # Секунды

        # --- ИСПРАВЛЕНИЕ: Используем chat.send_message_async(stream=True) ---
        response_stream = await chat.send_message_async(user_message, stream=True)

        async for chunk in response_stream:
            if chunk.text:
                full_response_text += chunk.text
                current_time = time.time()
                if current_time - last_update_time > update_interval:
                    try:
                        await placeholder_message.edit_text(full_response_text + " ✍️")
                        last_update_time = current_time
                    except telegram.error.BadRequest:
                        pass
        
        # Убираем индикатор "печатает" в финальном сообщении
        if placeholder_message and full_response_text:
            await placeholder_message.edit_text(full_response_text)
        
        # Обновляем историю после получения полного ответа. chat.history уже обновлен.
        update_history(user_id, chat.history)

    except Exception as e:
        logger.error(f"Ошибка при обработке текстового сообщения со стримингом: {e}")
        error_text = f'К сожалению, произошла ошибка: {e}'
        if placeholder_message: await placeholder_message.edit_text(error_text)
        else: await update.message.reply_text(error_text)


@restricted
async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # ... (код без изменений)

@restricted
async def handle_document_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # ... (код без изменений)


# --- Точка входа для сервера ---
def main() -> None:
    # ... (код без изменений)

if __name__ == "__main__":
    # ... (код без изменений)
