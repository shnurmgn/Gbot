import logging
import asyncio
import io
import os
import json
import google.generativeai as genai
from fastapi import FastAPI, Request, Response
import telegram
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from PIL import Image
import fitz
from upstash_redis import Redis

# --- Настройка ---
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
ALLOWED_USER_IDS_STR = os.environ.get('ALLOWED_USER_IDS')
ALLOWED_USER_IDS = [int(user_id.strip()) for user_id in ALLOWED_USER_IDS_STR.split(',')] if ALLOWED_USER_IDS_STR else []

# --- Инициализация клиентов (глобальные объекты) ---
# Создаем один экземпляр бота, который будет переиспользоваться
bot_instance = telegram.Bot(token=TELEGRAM_BOT_TOKEN)

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

# --- Вспомогательные функции ---
async def send_long_message(chat_id: int, text: str):
    if not text.strip(): return
    if len(text) <= 4096:
        await bot_instance.send_message(chat_id, text)
    else:
        for i in range(0, len(text), 4096):
            chunk = text[i:i + 4096]
            await bot_instance.send_message(chat_id, chunk)
            await asyncio.sleep(0.5)

async def handle_gemini_response(chat_id: int, response):
    try:
        if not response.candidates:
            await bot_instance.send_message(chat_id, f"⚠️ Запрос был заблокирован.")
            return
        candidate = response.candidates[0]
        if candidate.finish_reason.name != "STOP":
            await bot_instance.send_message(chat_id, f"⚠️ Контент не может быть сгенерирован. Причина: `{candidate.finish_reason.name}`", parse_mode='Markdown')
            return
        if not candidate.content.parts:
            await bot_instance.send_message(chat_id, "Модель вернула пустой ответ.")
            return
        for part in candidate.content.parts:
            if hasattr(part, 'text') and part.text:
                await send_long_message(chat_id, part.text)
            elif hasattr(part, 'inline_data') and part.inline_data.mime_type.startswith('image/'):
                await bot_instance.send_photo(chat_id, photo=io.BytesIO(part.inline_data.data))
    except Exception as e:
        logger.error(f"Критическая ошибка при обработке ответа от Gemini: {e}")
        await bot_instance.send_message(chat_id, f"Произошла критическая ошибка при обработке ответа: {e}")

def get_history(user_id: int) -> list:
    if not redis_client: return []
    try:
        history_data = redis_client.get(f"history:{user_id}")
        return json.loads(history_data) if history_data else []
    except Exception: return []

def update_history(user_id: int, chat_history: list):
    if not redis_client: return
    history_to_save = [{'role': p.role, 'parts': [part.text for part in p.parts]} for p in chat_history]
    if len(history_to_save) > 10:
        history_to_save = history_to_save[-10:]
    redis_client.set(f"history:{user_id}", json.dumps(history_to_save), ex=86400)

def get_user_model(user_id: int) -> str:
    default_model = 'gemini-1.5-flash'
    if not redis_client: return default_model
    try:
        stored_model = redis_client.get(f"user:{user_id}:model")
        return stored_model if stored_model else default_model
    except Exception: return default_model

# --- Логика обработки команд и сообщений ---
async def handle_update(update: Update):
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    if user_id not in ALLOWED_USER_IDS:
        logger.warning(f"Неавторизованный доступ отклонен для пользователя с ID: {user_id}")
        return

    if update.callback_query:
        query = update.callback_query
        await query.answer()
        selected_model = query.data
        if redis_client: redis_client.set(f"user:{user_id}:model", selected_model)
        await query.edit_message_text(text=f"Модель изменена на: {selected_model}. Я запомню ваш выбор.")
        return

    if not update.message: return

    if update.message.text:
        text = update.message.text
        if text == "/start":
            model_name = get_user_model(user_id)
            await bot_instance.send_message(chat_id, f"Привет! Я бот Gemini.\nТекущая модель: {model_name}.\nЧтобы очистить память, используйте /clear.")
        elif text == "/clear":
            if redis_client: redis_client.delete(f"history:{user_id}")
            await bot_instance.send_message(chat_id, "Память очищена.")
        elif text == "/model":
            keyboard = [[InlineKeyboardButton(name, callback_data=name)] for name in ['gemini-2.5-pro', 'gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-2.5-flash-image-preview']]
            await bot_instance.send_message(chat_id, 'Выберите модель:', reply_markup=InlineKeyboardMarkup(keyboard))
        else:
            await bot_instance.send_chat_action(chat_id, telegram.constants.ChatAction.TYPING)
            model_name = get_user_model(user_id)
            history = get_history(user_id)
            model = genai.GenerativeModel(model_name)
            chat = model.start_chat(history=history)
            response = await chat.send_message_async(text)
            update_history(user_id, chat.history)
            await handle_gemini_response(chat_id, response)
    
    elif update.message.photo:
        await bot_instance.send_chat_action(chat_id, telegram.constants.ChatAction.UPLOAD_PHOTO)
        model_name = get_user_model(user_id)
        if model_name != 'gemini-2.5-flash-image-preview':
            await bot_instance.send_message(chat_id, "Чтобы работать с фото, выберите модель 'Nano Banana' через /model.")
            return
        photo_file = await update.message.photo[-1].get_file()
        caption = update.message.caption or "Опиши это изображение"
        photo_bytes = io.BytesIO()
        await photo_file.download_to_memory(photo_bytes)
        photo_bytes.seek(0)
        img = Image.open(photo_bytes)
        model_gemini = genai.GenerativeModel(model_name)
        response = await model_gemini.generate_content_async([caption, img])
        await handle_gemini_response(chat_id, response)

    elif update.message.document and update.message.document.mime_type == 'application/pdf':
        await bot_instance.send_chat_action(chat_id, telegram.constants.ChatAction.TYPING)
        model_name = get_user_model(user_id)
        if model_name not in ['gemini-1.5-pro', 'gemini-2.5-pro']:
            await bot_instance.send_message(chat_id, f"Для анализа PDF, выберите модель Pro...")
            return
        doc_file = await update.message.document.get_file()
        caption = update.message.caption or "Проанализируй этот документ и сделай краткую выжимку."
        await bot_instance.send_message(chat_id, f"Получил PDF: {update.message.document.file_name}...")
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
        await bot_instance.send_message(chat_id, f"Отправляю первые {num_pages} страниц в Gemini на анализ...")
        model_gemini = genai.GenerativeModel(model_name)
        response = await model_gemini.generate_content_async(content_parts)
        await handle_gemini_response(chat_id, response)

# --- Точка входа для Vercel на FastAPI ---
app = FastAPI()

@app.post("/api/bot")
async def webhook(request: Request):
    """Асинхронная точка входа, которая вручную обрабатывает Update."""
    try:
        update_data = await request.json()
        update = Update.de_json(update_data, bot_instance)
        await handle_update(update)
        return Response(status_code=200)
    except Exception as e:
        logger.error(f"Ошибка в webhook FastAPI: {e}")
        return Response(status_code=500)

@app.get("/")
def index():
    return "Bot is running..."
