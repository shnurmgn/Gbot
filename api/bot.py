import logging
import asyncio
import io
import os
import json
import google.generativeai as genai
from fastapi import FastAPI, Request, Response, HTTPException
import telegram
from telegram import Update
from PIL import Image
import fitz
from upstash_redis import Redis

# --- Настройка ---
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
ALLOWED_USER_IDS_STR = os.environ.get('ALLOWED_USER_IDS')
ALLOWED_USER_IDS = [int(user_id.strip()) for user_id in ALLOWED_USER_IDS_STR.split(',')] if ALLOWED_USER_IDS_STR else []
CRON_SECRET = os.environ.get('CRON_SECRET') # Секретный ключ для Cron
MESSAGE_QUEUE_KEY = "message_queue"

# --- Подключение к Upstash Redis ---
redis_client = None
try:
    redis_client = Redis(
        url=os.environ.get('UPSTASH_REDIS_URL'),
        token=os.environ.get('UPSTASH_REDIS_TOKEN'),
        decode_responses=True # Важно для работы со строками
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

# --- ВАЖНО: Весь наш старый код (обработчики, хелперы) теперь внутри одной асинхронной функции ---
async def process_single_update(update_json: str):
    """
    Эта функция содержит всю логику нашего бота.
    Она будет вызываться из Cron Job для каждого сообщения из очереди.
    """
    bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
    update = Update.de_json(json.loads(update_json), bot)
    
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    # 1. Авторизация
    if user_id not in ALLOWED_USER_IDS:
        logger.warning(f"Неавторизованный доступ отклонен для пользователя с ID: {user_id}")
        return # Тихо игнорируем

    # 2. ПРОСТОЙ ПРИМЕР: отвечаем "эхом" на текстовое сообщение
    #    СЮДА НУЖНО БУДЕТ ПОСТЕПЕННО ПЕРЕНЕСТИ ВСЮ ВАШУ ЛОГИКУ ОБРАБОТКИ
    if update.message and update.message.text:
        # Здесь должна быть ваша логика вызова Gemini
        # Например:
        # model = genai.GenerativeModel('gemini-1.5-flash')
        # response = await model.generate_content_async(update.message.text)
        # await bot.send_message(chat_id=chat_id, text=response.text)
        
        # Для начала, простое эхо:
        await bot.send_message(
            chat_id=chat_id,
            text=f"Сообщение получено и обработано фоновым процессом: '{update.message.text}'"
        )
    else:
        await bot.send_message(chat_id=chat_id, text="Получено нетекстовое сообщение.")


# --- Точка входа для Vercel на FastAPI ---
app = FastAPI()

@app.post("/api/bot")
async def webhook_receiver(request: Request):
    """
    ПРИЕМНИК: Мгновенно принимает сообщение от Telegram и кладет его в очередь (Redis).
    """
    try:
        update_data = await request.json()
        if redis_client:
            # Кладем сообщение в очередь
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
    if request.query_params.get('cron_secret') != CRON_SECRET:
        logger.warning("Попытка несанкционированного вызова Cron Job.")
        raise HTTPException(status_code=401, detail="Unauthorized")

    # 2. Обрабатываем очередь
    if not redis_client:
        logger.error("CRON: Redis не подключен, обработка невозможна.")
        return Response(status_code=500)
    
    try:
        # Забираем до 5 сообщений из очереди за раз
        # lpop возвращает одно, нужно будет использовать пайплайн для нескольких
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

@app.get("/")
def index():
    return "Bot webhook receiver is running..."
