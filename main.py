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

# --- Настройка ---
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
ALLOWED_USER_IDS_STR = os.environ.get('ALLOWED_USER_IDS')
ALLOWED_USER_IDS = [int(user_id.strip()) for user_id in ALLOWED_USER_IDS_STR.split(',')] if ALLOWED_USER_IDS_STR else []

TELEGRAM_MAX_MESSAGE_LENGTH = 4096
DOCUMENT_ANALYSIS_MODELS = ['gemini-1.5-pro', 'gemini-2.5-pro']
HISTORY_LIMIT = 10 

# --- Подключение к Upstash Redis ---
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

# --- Настройка логирования и Gemini API ---
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# --- Декоратор для проверки авторизации ---
def restricted(func):
    @wraps(func)
    async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        user_id = update.effective_user.id
        if user_id not in ALLOWED_USER_IDS:
            if update.message: await update.message.reply_text("⛔️ У вас нет доступа к этому боту.")
            return
        return await func(update, context, *args, **kwargs)
    return wrapped

# --- Вспомогательные функции ---

async def handle_gemini_response_stream(update: Update, response_stream):
    """Обрабатывает потоковый ответ от Gemini, редактируя сообщение в реальном времени."""
    placeholder_message = None
    full_response_text = ""
    last_update_time = 0
    update_interval = 0.7  # Секунды

    try:
        # Отправляем начальное сообщение-заготовку
        placeholder_message = await update.message.reply_text("...")
        last_update_time = time.time()

        async for chunk in response_stream:
            if chunk.text:
                full_response_text += chunk.text
                current_time = time.time()
                if current_time - last_update_time > update_interval:
                    try:
                        await placeholder_message.edit_text(full_response_text + " ✍️")
                        last_update_time = current_time
                    except telegram.error.BadRequest: # Игнорируем ошибку, если текст не изменился
                        pass
        
        # Убираем индикатор "печатает" в финальном сообщении
        if placeholder_message and full_response_text:
            await placeholder_message.edit_text(full_response_text)
        
        # Обновляем историю после получения полного ответа
        update_history(update.effective_user.id, update.message.text, full_response_text)

    except Exception as e:
        logger.error(f"Критическая ошибка при обработке стриминг-ответа от Gemini: {e}")
        if placeholder_message:
            await placeholder_message.edit_text(f"Произошла ошибка при генерации ответа: {e}")
        else:
            await update.message.reply_text(f"Произошла ошибка при генерации ответа: {e}")


def get_history(user_id: int) -> list:
    if not redis_client: return []
    try:
        history_data = redis_client.get(f"history:{user_id}")
        return json.loads(history_data) if history_data else []
    except Exception: return []

def update_history(user_id: int, user_message_text: str, model_response_text: str):
    if not redis_client: return
    history = get_history(user_id)
    history.append({'role': 'user', 'parts': [{'text': user_message_text}]})
    history.append({'role': 'model', 'parts': [{'text': model_response_text}]})
    if len(history) > HISTORY_LIMIT:
        history = history[-HISTORY_LIMIT:]
    redis_client.set(f"history:{user_id}", json.dumps(history), ex=86400)

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
    # ... (код без изменений)

@restricted
async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # ... (код без изменений)

@restricted
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_message = update.message.text
    model_name = get_user_model(user_id)
    await update.message.reply_chat_action(telegram.constants.ChatAction.TYPING)
    try:
        history = get_history(user_id)
        model = genai.GenerativeModel(model_name)
        
        # Формируем контент для отправки, включая историю
        content_with_history = [
            {'role': h['role'], 'parts': h['parts']} for h in history
        ]
        content_with_history.append({'role': 'user', 'parts': [{'text': user_message}]})

        # Запускаем генерацию в режиме stream=True
        response_stream = await model.generate_content_async(content_with_history, stream=True)
        
        # Передаем стрим в новый обработчик
        await handle_gemini_response_stream(update, response_stream)
        
    except Exception as e:
        logger.error(f"Ошибка при обработке текстового сообщения со стримингом: {e}")
        await update.message.reply_text(f'К сожалению, произошла ошибка: {e}')

@restricted
async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # ... (вставьте сюда полный код функции handle_photo_message из предыдущего ответа)

@restricted
async def handle_document_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # ... (вставьте сюда полный код функции handle_document_message из предыдущего ответа)

# --- Точка входа для сервера ---
def main() -> None:
    # ... (код без изменений)
    
if __name__ == "__main__":
    # ... (код без изменений)
