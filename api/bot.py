import logging
import asyncio
import io
import os
import json
import google.generativeai as genai
from fastapi import FastAPI, Request, Response, HTTPException
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
CRON_SECRET = os.environ.get('CRON_SECRET')

TELEGRAM_MAX_MESSAGE_LENGTH = 4096
DOCUMENT_ANALYSIS_MODELS = ['gemini-1.5-pro', 'gemini-2.5-pro']
HISTORY_LIMIT = 10
MESSAGE_QUEUE_KEY = "message_queue"

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
    logging.error(f"Не удалось подключиться к Redis: {e}")

# --- Настройка логирования и Gemini API ---
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# --- Вспомогательные функции ---
# (Эти функции остаются почти без изменений, но теперь принимают объект 'bot')
async def send_long_message(bot: telegram.Bot, chat_id: int, text: str):
    if not text.strip(): return
    if len(text) <= TELEGRAM_MAX_MESSAGE_LENGTH:
        await bot.send_message(chat_id, text)
    else:
        for i in range(0, len(text), TELEGRAM_MAX_MESSAGE_LENGTH):
            chunk = text[i:i + TELEGRAM_MAX_MESSAGE_LENGTH]
            await bot.send_message(chat_id, chunk)
            await asyncio.sleep(0.5)

async def handle_gemini_response(bot: telegram.Bot, chat_id: int, response):
    try:
        if not response.candidates:
            prompt_feedback = response.prompt_feedback
            block_reason = getattr(prompt_feedback, 'block_reason_message', 'Причина не указана.')
            await bot.send_message(chat_id, f"⚠️ Запрос был заблокирован.\nПричина: {block_reason}")
            return
        candidate = response.candidates[0]
        if candidate.finish_reason.name != "STOP":
            finish_reason_name = candidate.finish_reason.name
            safety_info = []
            for rating in candidate.safety_ratings:
                if rating.probability.name in ["MEDIUM", "HIGH"]:
                     safety_info.append(f"- Категория: {rating.category.name.replace('HARM_CATEGORY_', '')}, Вероятность: {rating.probability.name}")
            details = "\n".join(safety_info)
            await bot.send_message(
                chat_id,
                f"⚠️ Контент не может быть сгенерирован.\n\n"
                f"**Причина:** `{finish_reason_name}`\n"
                f"**Детали:**\n{details if details else 'Нет дополнительных деталей.'}",
                parse_mode='Markdown'
            )
            return
        if not candidate.content.parts:
            await bot.send_message(chat_id, "Модель вернула пустой ответ.")
            return
        for part in candidate.content.parts:
            if hasattr(part, 'text') and part.text:
                await send_long_message(bot, chat_id, part.text)
            elif hasattr(part, 'inline_data') and part.inline_data.mime_type.startswith('image/'):
                image_data = part.inline_data.data
                await bot.send_photo(chat_id, photo=io.BytesIO(image_data))
    except Exception as e:
        logger.error(f"Критическая ошибка при обработке ответа от Gemini: {e}")
        await bot.send_message(chat_id, f"Произошла критическая ошибка при обработке ответа от модели: {e}")

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

# --- ГЛАВНАЯ ФУНКЦИЯ-ОБРАБОТЧИК ---
async def process_single_update(update_json: str):
    bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
    update = Update.de_json(json.loads(update_json), bot)
    
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    # 1. Авторизация
    if user_id not in ALLOWED_USER_IDS:
        logger.warning(f"Неавторизованный доступ отклонен для пользователя с ID: {user_id}")
        # Тихо игнорируем, чтобы не отвечать неавторизованным пользователям
        return

    # 2. Диспетчер (определяем тип сообщения)
    if update.callback_query:
        # Логика для кнопок
        query = update.callback_query
        await query.answer()
        selected_model = query.data
        if redis_client:
            redis_client.set(f"user:{user_id}:model", selected_model)
        message_text = f"Модель изменена на: {selected_model}. Я запомню ваш выбор."
        await query.edit_message_text(text=message_text)

    elif update.message:
        if update.message.text:
            # Логика для текстовых команд и сообщений
            text = update.message.text
            if text == "/start":
                model_name = get_user_model(user_id)
                await bot.send_message(chat_id, f"Привет! Я бот Gemini.\nТекущая модель: {model_name}.\nЧтобы очистить память, используйте /clear.")
            elif text == "/clear":
                if redis_client: redis_client.delete(f"history:{user_id}")
                await bot.send_message(chat_id, "Память очищена.")
            elif text == "/model":
                keyboard = [[InlineKeyboardButton(name, callback_data=name)] for name in ['gemini-2.5-pro', 'gemini-1.5-pro', 'gemini-1.5-flash']]
                await bot.send_message(chat_id, 'Выберите модель:', reply_markup=InlineKeyboardMarkup(keyboard))
            else:
                # Обычное текстовое сообщение с историей
                await bot.send_chat_action(chat_id, telegram.constants.ChatAction.TYPING)
                model_name = get_user_model(user_id)
                history = get_history(user_id)
                model = genai.GenerativeModel(model_name)
                chat = model.start_chat(history=history)
                response = await chat.send_message_async(text)
                update_history(user_id, chat.history)
                await handle_gemini_response(bot, chat_id, response)
        
        elif update.message.photo:
            # Логика для фото
            await bot.send_chat_action(chat_id, telegram.constants.ChatAction.UPLOAD_PHOTO)
            photo_file = await update.message.photo[-1].get_file()
            caption = update.message.caption or "Опиши это изображение"
            photo_bytes = io.BytesIO()
            await photo_file.download_to_memory(photo_bytes)
            photo_bytes.seek(0)
            img = Image.open(photo_bytes)
            model = genai.GenerativeModel('gemini-1.5-flash-preview') # Упрощено для примера
            response = model.generate_content([caption, img])
            await handle_gemini_response(bot, chat_id, response)

        elif update.message.document and update.message.document.mime_type == 'application/pdf':
            # Логика для PDF
            # (Эту часть можно будет добавить по аналогии, если потребуется)
            await bot.send_message(chat_id, "Обработка PDF в этой архитектуре пока не реализована.")

# --- Точка входа для Vercel на FastAPI ---
app = FastAPI()

@app.post("/api/bot")
async def webhook_receiver(request: Request):
    """ПРИЕМНИК: Мгновенно принимает сообщение от Telegram и кладет его в очередь."""
    try:
        update_data = await request.json()
        if redis_client:
            redis_client.rpush(MESSAGE_QUEUE_KEY, json.dumps(update_data))
            return Response(status_code=200)
        else:
            logger.error("WEBHOOK: Redis не подключен, сообщение не сохранено.")
            return Response(status_code=500)
    except Exception as e:
        logger.error(f"Ошибка в webhook_receiver: {e}")
        return Response(status_code=500)

@app.get("/api/process-queue")
async def cron_handler(request: Request):
    """ОБРАБОТЧИК: Вызывается Vercel Cron Job каждую минуту."""
    # 1. Проверяем секретный ключ
    cron_secret_header = request.headers.get('x-vercel-cron-secret')
    if cron_secret_header != CRON_SECRET:
        logger.warning("Попытка несанкционированного вызова Cron Job.")
        raise HTTPException(status_code=401, detail="Unauthorized")

    # 2. Обрабатываем очередь
    if not redis_client:
        logger.error("CRON: Redis не подключен, обработка невозможна.")
        return Response(status_code=500)
    
    try:
        # Забираем до 5 сообщений из очереди за раз
        messages_to_process = redis_client.lpop(MESSAGE_QUEUE_KEY, 5)
        if not messages_to_process:
            return Response(content='Очередь пуста.', status_code=200)
        
        # Запускаем обработку параллельно
        tasks = [process_single_update(msg) for msg in messages_to_process]
        await asyncio.gather(*tasks)
        
        return Response(content=f'Обработано {len(messages_to_process)} сообщений.', status_code=200)
    except Exception as e:
        logger.error(f"CRON: Критическая ошибка при обработке очереди: {e}")
        return Response(status_code=500)
