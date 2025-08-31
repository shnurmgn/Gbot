import logging
import asyncio
import io
import os
from functools import wraps
import json
import google.generativeai as genai
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
        token=os.environ.get('UPSTASH_REDIS_TOKEN')
    )
    redis_client.ping()
    logging.info("Успешно подключено к Upstash Redis.")
except Exception as e:
    logging.error(f"Не удалось подключиться к Redis. Убедитесь, что UPSTASH_REDIS_URL и UPSTASH_REDIS_TOKEN правильно заданы в Vercel. Ошибка: {e}")

# --- Настройка логирования ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Инициализация Gemini API ---
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# --- Декоратор для проверки авторизации ---
def restricted(func):
    @wraps(func)
    async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        user_id = update.effective_user.id
        if user_id not in ALLOWED_USER_IDS:
            if update.message: await update.message.reply_text("⛔️ У вас нет доступа к этому боту.")
            elif update.callback_query: await update.callback_query.answer("⛔️ У вас нет доступа к этому боту.", show_alert=True)
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
            await update.message.reply_text(f"⚠️ Запрос был заблокирован.\nПричина: {block_reason}")
            return

        candidate = response.candidates[0]
        if candidate.finish_reason.name != "STOP":
            finish_reason_name = candidate.finish_reason.name
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
            await update.message.reply_text("Модель вернула пустой ответ.")
            return

        for part in candidate.content.parts:
            if hasattr(part, 'text') and part.text:
                await send_long_message(update, part.text)
            elif hasattr(part, 'inline_data') and part.inline_data.mime_type.startswith('image/'):
                image_data = part.inline_data.data
                await update.message.reply_photo(photo=io.BytesIO(image_data))
    except Exception as e:
        logger.error(f"Критическая ошибка при обработке ответа от Gemini: {e}")
        await update.message.reply_text(f"Произошла критическая ошибка при обработке ответа от модели: {e}")

def get_history(user_id: int) -> list:
    if not redis_client: return []
    try:
        history_data = redis_client.get(f"history:{user_id}")
        return json.loads(history_data) if history_data else []
    except Exception as e:
        logger.error(f"Ошибка чтения истории из Redis для user_id {user_id}: {e}")
        return []

def update_history(user_id: int, chat_history: list):
    if not redis_client: return
    history_to_save = [{'role': p.role, 'parts': [part.text for part in p.parts]} for p in chat_history]
    if len(history_to_save) > HISTORY_LIMIT:
        history_to_save = history_to_save[-HISTORY_LIMIT:]
    try:
        redis_client.set(f"history:{user_id}", json.dumps(history_to_save), ex=86400)
    except Exception as e:
        logger.error(f"Ошибка сохранения истории в Redis для user_id {user_id}: {e}")

def get_user_model(user_id: int) -> str:
    default_model = 'gemini-1.5-flash'
    if not redis_client: return default_model
    try:
        stored_model = redis_client.get(f"user:{user_id}:model")
        return stored_model if stored_model else default_model
    except Exception as e:
        logger.error(f"Ошибка чтения модели из Redis для user_id {user_id}: {e}")
        return default_model

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
    if not redis_client:
        await update.message.reply_text("Хранилище не подключено.")
        return
    try:
        redis_client.delete(f"history:{user_id}")
        logger.info(f"История для пользователя {user_id} очищена.")
        await update.message.reply_text("Память очищена. Начинаем новый диалог!")
    except Exception as e:
        logger.error(f"Ошибка очистки истории в Redis для user_id {user_id}: {e}")
        await update.message.reply_text("Не удалось очистить историю.")

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
    user_id = query.from_user.id
    await query.answer()
    selected_model = query.data
    
    if not redis_client:
        await query.edit_message_text(text="Хранилище данных не настроено.")
        return

    try:
        redis_client.set(f"user:{user_id}:model", selected_model)
        message_text = f"Модель изменена на: {selected_model}. Я запомню ваш выбор."
        if selected_model in DOCUMENT_ANALYSIS_MODELS:
            message_text += "\n\nЭта модель отлично подходит для анализа PDF."
        await query.edit_message_text(text=message_text)
    except Exception as e:
        logger.error(f"Ошибка сохранения модели в Redis: {e}")
        await query.edit_message_text(text="Не удалось сохранить ваш выбор.")

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
        content_parts = [caption, img]
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(content_parts)
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

application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

application.add_handler(CommandHandler("start", start))
application.add_handler(CommandHandler("clear", clear_history))
application.add_handler(CommandHandler("model", model_selection))
application.add_handler(CallbackQueryHandler(button_callback))
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
application.add_handler(MessageHandler(filters.PHOTO, handle_photo_message))
application.add_handler(MessageHandler(filters.Document.PDF, handle_document_message))

app = Flask(__name__)

async def process_update_async(update_data):
    async with application:
        update = Update.de_json(update_data, application.bot)
        await application.process_update(update)

@app.route('/api/bot', methods=['POST'])
def webhook():
    """
    Финальная, каноническая версия точки входа для Vercel.
    """
    try:
        asyncio.run(process_update_async(request.get_json(force=True)))
        return Response('ok', status=200)
    except Exception as e:
        logger.error(f"Ошибка в webhook: {e}")
        return Response('error', status=500)
