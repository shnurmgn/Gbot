import logging
import asyncio
import io
import os
from functools import wraps
import json
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

# --- Настройка (читает переменные окружения сервера) ---
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
    """Декоратор для ограничения доступа к боту."""
    @wraps(func)
    async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        user_id = update.effective_user.id
        if user_id not in ALLOWED_USER_IDS:
            logger.warning(f"Неавторизованный доступ отклонен для пользователя с ID: {user_id}")
            if update.message: await update.message.reply_text("⛔️ У вас нет доступа к этому боту.")
            return
        return await func(update, context, *args, **kwargs)
    return wrapped

# --- Вспомогательные функции ---
async def send_long_message(update: Update, text: str):
    if not text.strip(): return
    if len(text) <= TELEGRAM_MAX_MESSAGE_LENGTH:
        await update.message.reply_text(text)
    else:
        for i in range(0, len(text), TELEGRAM_MAX_MESSAGE_LENGTH):
            await update.message.reply_text(text[i:i + TELEGRAM_MAX_MESSAGE_LENGTH])
            await asyncio.sleep(0.5)

async def handle_gemini_response(update: Update, response):
    try:
        if not response.candidates:
            await update.message.reply_text(f"⚠️ Запрос был заблокирован.\nПричина: {getattr(response.prompt_feedback, 'block_reason_message', 'Причина не указана.')}")
            return
        candidate = response.candidates[0]
        if candidate.finish_reason.name != "STOP":
            await update.message.reply_text(f"⚠️ Контент не может быть сгенерирован. Причина: `{candidate.finish_reason.name}`", parse_mode='Markdown')
            return
        if not candidate.content.parts:
            await update.message.reply_text("Модель вернула пустой ответ.")
            return
        for part in candidate.content.parts:
            if hasattr(part, 'text') and part.text:
                await send_long_message(update, part.text)
            elif hasattr(part, 'inline_data') and part.inline_data.mime_type.startswith('image/'):
                await update.message.reply_photo(photo=io.BytesIO(part.inline_data.data))
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
    await update.message.reply_chat_action(telegram.constants.ChatAction.TYPING)
    try:
        history = get_history(user_id)
        model = genai.GenerativeModel(model_name)
        chat = model.start_chat(history=history)
        response = await chat.send_message_async(user_message)
        update_history(user_id, chat.history)
        await handle_gemini_response(update, response)
    except Exception as e:
        logger.error(f"Ошибка при обработке текстового сообщения с историей: {e}")
        await update.message.reply_text(f'К сожалению, произошла ошибка: {e}')

@restricted
async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    model_name = get_user_model(user_id)
    if model_name != 'gemini-2.5-flash-image-preview':
        await update.message.reply_text("Чтобы работать с фото, выберите модель 'Nano Banana' через /model.")
        return
    photo_file = await update.message.photo[-1].get_file()
    caption = update.message.caption or "Опиши это изображение"
    await update.message.reply_chat_action(telegram.constants.ChatAction.UPLOAD_PHOTO)
    try:
        photo_bytes = io.BytesIO()
        await photo_file.download_to_memory(photo_bytes)
        photo_bytes.seek(0)
        img = Image.open(photo_bytes)
        model_gemini = genai.GenerativeModel(model_name)
        response = await model_gemini.generate_content_async([caption, img])
        await handle_gemini_response(update, response)
    except Exception as e:
        logger.error(f"Ошибка при обработке фото: {e}")
        await update.message.reply_text(f'К сожалению, произошла ошибка при обработке фото: {e}')

@restricted
async def handle_document_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    model_name = get_user_model(user_id)
    if model_name not in DOCUMENT_ANALYSIS_MODELS:
        await update.message.reply_text(f"Для анализа PDF, пожалуйста, выберите модель Pro...")
        return
    doc_file = await update.message.document.get_file()
    caption = update.message.caption or "Проанализируй этот документ и сделай краткую выжимку."
    await update.message.reply_text(f"Получил PDF: {update.message.document.file_name}...")
    try:
        pdf_bytes = io.BytesIO()
        await doc_file.download_to_memory(pdf_bytes)
        pdf_bytes.seek(0)
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        content_parts = [caption]
        page_limit = 25 
        num_pages = min(len(pdf_document), page_limit)
        for page_num in range(num_pages):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap()
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))
            content_parts.append(img)
        pdf_document.close()
        await update.message.reply_text(f"Отправляю первые {num_pages} страниц в Gemini на анализ...")
        model_gemini = genai.GenerativeModel(model_name)
        response = await model_gemini.generate_content_async(content_parts)
        await handle_gemini_response(update, response)
    except Exception as e:
        logger.error(f"Ошибка при обработке PDF: {e}")
        await update.message.reply_text(f'К сожалению, произошла ошибка при обработке PDF: {e}')

# --- Точка входа для постоянной работы на сервере ---
def main() -> None:
    """Запускает бота в режиме polling."""
    logger.info("Создание и настройка приложения...")
    
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Регистрация всех наших обработчиков
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("clear", clear_history))
    application.add_handler(CommandHandler("model", model_selection))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo_message))
    application.add_handler(MessageHandler(filters.Document.PDF, handle_document_message))

    # Запуск бота. Он будет работать, пока вы не остановите процесс (Ctrl+C).
    logger.info("Бот запущен и работает в режиме опроса...")
    application.run_polling()

if __name__ == "__main__":
    if not all([TELEGRAM_BOT_TOKEN, GEMINI_API_KEY, ALLOWED_USER_IDS_STR, redis_client]):
        logger.critical("Не все переменные окружения или подключения настроены! Бот не может запуститься.")
    else:
        main()
