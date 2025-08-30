import logging
import asyncio
import io
import os
from functools import wraps
import google.generativeai as genai
from google.generativeai import protos
from flask import Flask, request, Response
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

# --- Настройка ---
# Токены и ID берутся из переменных окружения Vercel для безопасности
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', 'YOUR_FALLBACK_TOKEN')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'YOUR_FALLBACK_API_KEY')
# ID пользователей указываются через запятую в настройках Vercel
ALLOWED_USER_IDS_STR = os.environ.get('ALLOWED_USER_IDS', '123456789')
ALLOWED_USER_IDS = [int(user_id.strip()) for user_id in ALLOWED_USER_IDS_STR.split(',')]

TELEGRAM_MAX_MESSAGE_LENGTH = 4096
DOCUMENT_ANALYSIS_MODELS = ['gemini-1.5-pro', 'gemini-2.5-pro']

# --- Настройка логирования ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# --- Инициализация Gemini API ---
try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    logger.error(f"Ошибка конфигурации Gemini API: {e}")
    # В среде Vercel мы не можем просто выйти, поэтому просто логируем
    
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
                await update.callback_query.answer("⛔️ У вас нет доступа к этому боту.", show_alert=True)
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
            chunk = text[i:i + TELEGRAM_MAX_MESSAGE_LENGTH]
            await update.message.reply_text(chunk)
            await asyncio.sleep(0.5)

async def handle_gemini_response(update: Update, response):
    try:
        if not response.candidates:
            prompt_feedback = response.prompt_feedback
            block_reason = getattr(prompt_feedback, 'block_reason_message', 'Причина не указана.')
            logger.warning(f"Запрос полностью заблокирован. Причина: {block_reason}. Полный ответ: {response}")
            await update.message.reply_text(f"⚠️ Запрос был заблокирован.\nПричина: {block_reason}")
            return

        candidate = response.candidates[0]
        if candidate.finish_reason.name != "STOP":
            finish_reason_name = candidate.finish_reason.name
            logger.warning(f"Генерация остановлена по причине: {finish_reason_name}. Полный ответ: {response}")
            safety_info = []
            for rating in candidate.safety_ratings:
                if rating.probability.name in ["MEDIUM", "HIGH"]:
                     safety_info.append(f"- Категория: {rating.category.name.replace('HARM_CATEGORY_', '')}, Вероятность: {rating.probability.name}")
            details = "\n".join(safety_info)
            await update.message.reply_text(
                f"⚠️ Контент не может быть сгенерирован.\n\n"
                f"**Причина:** `{finish_reason_name}`\n"
                f"**Детали:**\n{details if details else 'Нет дополнительных деталей.'}",
                parse_mode='Markdown'
            )
            return

        if not candidate.content.parts:
            logger.warning(f"Gemini вернул пустой ответ без контента. Полный ответ: {response}")
            await update.message.reply_text("Модель вернула пустой ответ.")
            return

        for part in candidate.content.parts:
            if hasattr(part, 'text') and part.text:
                await send_long_message(update, part.text)
            elif hasattr(part, 'inline_data') and part.inline_data.mime_type.startswith('image/'):
                image_data = part.inline_data.data
                await update.message.reply_photo(photo=io.BytesIO(image_data))
            else:
                logger.warning(f"Получена неизвестная часть ответа: {part}")
    except Exception as e:
        logger.error(f"Критическая ошибка при обработке ответа от Gemini: {e}")
        await update.message.reply_text(f"Произошла критическая ошибка при обработке ответа от модели: {e}")

# --- Функции-обработчики ---

@restricted
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    context.user_data.setdefault('model_name', 'gemini-1.5-flash')
    await update.message.reply_html(rf"Привет, {user.mention_html()}!")
    await update.message.reply_text(f"Я бот, подключенный к Gemini...")

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
    await update.message.reply_text('Пожалуйста, выберите модель:', reply_markup=reply_markup)

@restricted
async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    selected_model = query.data
    context.user_data['model_name'] = selected_model
    message_text = f"Модель изменена на: {selected_model}"
    if selected_model == 'gemini-2.5-flash-image-preview':
        message_text = "Выбрана модель 'Nano Banana'..."
    elif selected_model in DOCUMENT_ANALYSIS_MODELS:
         message_text += ".\n\nЭта модель отлично подходит для анализа PDF..."
    await query.edit_message_text(text=message_text)

@restricted
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    model_name = context.user_data.get('model_name', 'gemini-1.5-flash')
    await update.message.reply_chat_action(telegram.constants.ChatAction.TYPING)
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(user_message)
        await handle_gemini_response(update, response)
    except Exception as e:
        logger.error(f"Ошибка при обработке текстового сообщения: {e}")
        await update.message.reply_text(f'К сожалению, произошла ошибка: {e}')

@restricted
async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    model_name = context.user_data.get('model_name', 'gemini-1.5-flash')
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
        content_parts = [caption, img]
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(content_parts)
        await handle_gemini_response(update, response)
    except Exception as e:
        logger.error(f"Ошибка при обработке фото: {e}")
        await update.message.reply_text(f'К сожалению, произошла ошибка при обработке фото: {e}')

@restricted
async def handle_document_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    model_name = context.user_data.get('model_name', 'gemini-1.5-flash')
    if model_name not in DOCUMENT_ANALYSIS_MODELS:
        await update.message.reply_text(f"Для анализа PDF, выберите модель Pro...")
        return
    doc_file = await update.message.document.get_file()
    caption = update.message.caption or "Проанализируй этот документ и сделай краткую выжимку."
    await update.message.reply_text(f"Получил PDF: {update.message.document.file_name}...")
    await update.message.reply_chat_action(telegram.constants.ChatAction.TYPING)
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
        
        notice = f"Отправляю первые {num_pages} страниц в Gemini на анализ..."
        if len(pdf_document) > page_limit:
            notice += f"\n(Документ слишком большой, анализ ограничен первыми {page_limit} страницами)"
        await update.message.reply_text(notice)

        model = genai.GenerativeModel(model_name)
        response = model.generate_content(content_parts)
        await handle_gemini_response(update, response)
    except Exception as e:
        logger.error(f"Ошибка при обработке PDF: {e}")
        await update.message.reply_text(f'К сожалению, произошла ошибка при обработке PDF: {e}')


# --- Точка входа для Vercel ---

# 1. Создаем объект приложения один раз при старте
application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

# 2. Регистрируем все наши обработчики, как и раньше
application.add_handler(CommandHandler("start", start))
application.add_handler(CommandHandler("model", model_selection))
application.add_handler(CallbackQueryHandler(button_callback))
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
application.add_handler(MessageHandler(filters.PHOTO, handle_photo_message))
application.add_handler(MessageHandler(filters.Document.PDF, handle_document_message))

# 3. Создаем Flask-приложение, которое будет точкой входа для Vercel
app = Flask(__name__)

@app.route('/api/bot', methods=['POST'])
def webhook():
    """Эта функция вызывается каждый раз, когда Telegram присылает сообщение."""
    try:
        update_data = request.get_json(force=True)
        update = Update.de_json(update_data, application.bot)
        
        # Запускаем обработку этого одного сообщения в асинхронном режиме
        asyncio.run(application.process_update(update))
        
        return Response('ok', status=200)
    except Exception as e:
        logger.error(f"Ошибка в webhook: {e}")
        return Response('error', status=500)
