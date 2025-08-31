import logging
import asyncio
import io
import os
from functools import wraps
import json
import docx # <-- НОВЫЙ ИМПОРТ
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

# --- Декоратор и все вспомогательные функции (без изменений) ---
# ... (скопируйте сюда полный код функций restricted, send_long_message, handle_gemini_response, get_history, update_history, get_user_model из предыдущего полного ответа)

# --- Функции-обработчики (с изменениями в handle_document_message) ---
@restricted
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # ... (код без изменений)

@restricted
async def clear_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # ... (код без изменений)

@restricted
async def model_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # ... (код без изменений)

@restricted
async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # ... (код без изменений)

@restricted
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # ... (код без изменений)

@restricted
async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # ... (код без изменений)

# --- ОБНОВЛЕННЫЙ ОБРАБОТЧИК ДОКУМЕНТОВ ---
@restricted
async def handle_document_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    model_name = get_user_model(user_id)
    if model_name not in DOCUMENT_ANALYSIS_MODELS:
        await update.message.reply_text(f"Для анализа документов, пожалуйста, выберите модель Pro (например, Gemini 2.5 Pro) через команду /model.")
        return

    doc = update.message.document
    caption = update.message.caption or "Проанализируй этот документ и сделай краткую выжимку."
    
    await update.message.reply_text(f"Получил файл: {doc.file_name}.\nНачинаю обработку...")
    await update.message.reply_chat_action(telegram.constants.ChatAction.TYPING)

    try:
        doc_file = await doc.get_file()
        file_bytes_io = io.BytesIO()
        await doc_file.download_to_memory(file_bytes_io)
        file_bytes_io.seek(0)
        
        content_parts = [caption]
        file_text_content = ""

        # Определяем тип файла и извлекаем текст
        if doc.mime_type == 'application/pdf':
            # Для PDF используем старый метод: конвертируем страницы в картинки
            pdf_document = fitz.open(stream=file_bytes_io.read(), filetype="pdf")
            page_limit = 25 
            num_pages = min(len(pdf_document), page_limit)
            for page_num in range(num_pages):
                page = pdf_document.load_page(page_num)
                pix = page.get_pixmap()
                img_bytes = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_bytes))
                content_parts.append(img)
            pdf_document.close()
            await update.message.reply_text(f"Отправляю первые {num_pages} страниц PDF в Gemini на анализ...")

        elif doc.mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document': # .docx
            document = docx.Document(file_bytes_io)
            for para in document.paragraphs:
                file_text_content += para.text + "\n"
            content_parts.append(file_text_content)
        
        elif doc.mime_type == 'text/plain': # .txt
            file_text_content = file_bytes_io.read().decode('utf-8')
            content_parts.append(file_text_content)
            
        else:
            await update.message.reply_text(f"Извините, я пока не поддерживаю файлы типа {doc.mime_type}.")
            return

        model = genai.GenerativeModel(model_name)
        response = await model.generate_content_async(content_parts)
        await handle_gemini_response(update, response)

    except Exception as e:
        logger.error(f"Ошибка при обработке документа: {e}")
        await update.message.reply_text(f'К сожалению, произошла ошибка при обработке документа: {e}')

# --- Точка входа для сервера (с обновленным MessageHandler) ---
def main() -> None:
    logger.info("Создание и настройка приложения...")
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Регистрация всех наших обработчиков
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("clear", clear_history))
    application.add_handler(CommandHandler("model", model_selection))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo_message))
    
    # --- ИЗМЕНЕНИЕ: Теперь обработчик ловит PDF, DOCX и TXT ---
    supported_files_filter = filters.Document.PDF | filters.Document.DOCX | filters.Document.TXT
    application.add_handler(MessageHandler(supported_files_filter, handle_document_message))

    logger.info("Бот запущен и работает в режиме опроса...")
    application.run_polling()

if __name__ == "__main__":
    if not all([TELEGRAM_BOT_TOKEN, GEMINI_API_KEY, ALLOWED_USER_IDS_STR, redis_client]):
        logger.critical("Не все переменные окружения или подключения настроены! Бот не может запуститься.")
    else:
        main()
