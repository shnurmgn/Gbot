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
import telegram
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, ReplyKeyboardRemove
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
IMAGE_GEN_MODELS = ['gemini-2.5-flash-image-preview']
HISTORY_LIMIT = 10 
DEFAULT_CHAT_NAME = "default"

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
def update_usage_stats(user_id: int, usage_metadata):
    if not redis_client or not hasattr(usage_metadata, 'total_token_count'): return
    try:
        total_tokens = usage_metadata.total_token_count
        today = datetime.utcnow().strftime('%Y-%m-%d')
        daily_key = f"usage:{user_id}:daily:{today}"
        redis_client.incrby(daily_key, total_tokens)
        redis_client.expire(daily_key, 86400 * 2)
        this_month = datetime.utcnow().strftime('%Y-%m')
        monthly_key = f"usage:{user_id}:monthly:{this_month}"
        redis_client.incrby(monthly_key, total_tokens)
        redis_client.expire(monthly_key, 86400 * 32)
    except Exception as e:
        logger.error(f"Ошибка обновления статистики использования: {e}")

async def send_long_message(update: Update, text: str):
    if not text.strip(): return
    if len(text) <= TELEGRAM_MAX_MESSAGE_LENGTH:
        await update.message.reply_text(text)
    else:
        for i in range(0, len(text), TELEGRAM_MAX_MESSAGE_LENGTH):
            await update.message.reply_text(text[i:i + TELEGRAM_MAX_MESSAGE_LENGTH])
            await asyncio.sleep(0.5)

async def handle_gemini_response(update: Update, response):
    if hasattr(response, 'usage_metadata'):
        update_usage_stats(update.effective_user.id, response.usage_metadata)
    try:
        if not response.candidates:
            await update.message.reply_text(f"⚠️ Запрос был заблокирован.\nПричина: {getattr(response.prompt_feedback, 'block_reason_message', 'Причина не указана.')}")
            return
        candidate = response.candidates[0]
        if candidate.finish_reason.name != "STOP":
            await update.message.reply_text(f"⚠️ Контент не может быть сгенерирован. Причина: `{candidate.finish_reason.name}`", parse_mode='Markdown')
            return
        if not candidate.content.parts:
            await update.message.reply_text("Модель завершила работу, но не сгенерировала ответ. Попробуйте переформулировать ваш запрос.")
            return
        full_text = ""
        image_sent = False
        for part in candidate.content.parts:
            if hasattr(part, 'text') and part.text:
                full_text += part.text
            elif hasattr(part, 'inline_data') and part.inline_data.mime_type.startswith('image/'):
                await update.message.reply_photo(photo=io.BytesIO(part.inline_data.data))
                image_sent = True
        if full_text and not image_sent:
            await send_long_message(update, full_text)
    except Exception as e:
        logger.error(f"Критическая ошибка при обработке ответа от Gemini: {e}")
        await update.message.reply_text(f"Произошла критическая ошибка при обработке ответа: {e}")

async def handle_gemini_response_stream(update: Update, response_stream, user_message_text: str):
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
                        if len(full_response_text) < TELEGRAM_MAX_MESSAGE_LENGTH - 10:
                            await placeholder_message.edit_text(full_response_text + " ✍️")
                            last_update_time = current_time
                    except telegram.error.BadRequest:
                        pass
        if placeholder_message:
            await placeholder_message.delete()
        if not full_response_text.strip():
             await update.message.reply_text("Модель завершила работу, но не сгенерировала ответ. Попробуйте переформулировать ваш запрос.")
             return
        await send_long_message(update, full_response_text)
        update_history(update.effective_user.id, user_message_text, full_response_text)
        if hasattr(response_stream, 'usage_metadata') and response_stream.usage_metadata:
            update_usage_stats(update.effective_user.id, response_stream.usage_metadata)
    except Exception as e:
        logger.error(f"Критическая ошибка при обработке стриминг-ответа от Gemini: {e}")
        if placeholder_message: await placeholder_message.delete()
        await update.message.reply_text(f"Произошла ошибка при генерации ответа: {e}")

def get_active_chat_name(user_id: int) -> str:
    if not redis_client: return DEFAULT_CHAT_NAME
    return redis_client.get(f"active_chat:{user_id}") or DEFAULT_CHAT_NAME

def get_history(user_id: int) -> list:
    if not redis_client: return []
    active_chat = get_active_chat_name(user_id)
    try:
        history_data = redis_client.get(f"history:{user_id}:{active_chat}")
        return json.loads(history_data) if history_data else []
    except Exception: return []

def update_history(user_id: int, user_message_text: str, model_response_text: str):
    if not redis_client: return
    active_chat = get_active_chat_name(user_id)
    history = get_history(user_id)
    history.append({'role': 'user', 'parts': [{'text': user_message_text}]})
    history.append({'role': 'model', 'parts': [{'text': model_response_text}]})
    if len(history) > HISTORY_LIMIT:
        history = history[-HISTORY_LIMIT:]
    redis_client.set(f"history:{user_id}:{active_chat}", json.dumps(history), ex=86400 * 7)

def get_user_model(user_id: int) -> str:
    default_model = 'gemini-1.5-flash'
    if not redis_client: return default_model
    try:
        stored_model = redis_client.get(f"user:{user_id}:model")
        return stored_model if stored_model else default_model
    except Exception: return default_model

def get_user_persona(user_id: int) -> str:
    if not redis_client: return None
    return redis_client.get(f"persona:{user_id}")

# --- Функции-обработчики ---
@restricted
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    model_name = get_user_model(user.id)
    active_chat = get_active_chat_name(user.id)
    await update.message.reply_html(rf"Привет, {user.mention_html()}!")
    await update.message.reply_text(
        f"Я бот, подключенный к Gemini.\n"
        f"Текущая модель: `{model_name}`\n"
        f"Текущий чат: `{active_chat}`",
        parse_mode='Markdown'
    )
    await menu_command(update, context)

@restricted
async def menu_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        ["/model", "/persona"],
        ["/chats", "/new_chat", "/clear"],
        ["/usage", "/hide_menu"]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    await update.message.reply_text("📋 Главное меню:", reply_markup=reply_markup)

@restricted
async def hide_menu_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Меню скрыто.", reply_markup=ReplyKeyboardRemove())

@restricted
async def clear_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    active_chat = get_active_chat_name(user_id)
    if redis_client: redis_client.delete(f"history:{user_id}:{active_chat}")
    await update.message.reply_text(f"Память текущего чата (`{active_chat}`) очищена.")
    
@restricted
async def usage_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not redis_client:
        await update.message.reply_text("Хранилище не подключено, статистика недоступна.")
        return
    today = datetime.utcnow().strftime('%Y-%m-%d')
    this_month = datetime.utcnow().strftime('%Y-%m')
    daily_tokens = redis_client.get(f"usage:{user_id}:daily:{today}") or 0
    monthly_tokens = redis_client.get(f"usage:{user_id}:monthly:{this_month}") or 0
    await update.message.reply_text(
        f"📊 **Статистика использования токенов:**\n\n"
        f"Сегодня ({today}):\n`{int(daily_tokens):,}` токенов\n\n"
        f"В этом месяце ({this_month}):\n`{int(monthly_tokens):,}` токенов",
        parse_mode='Markdown'
    )

@restricted
async def persona_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    persona_text = " ".join(context.args) if context.args else None
    if not redis_client:
        await update.message.reply_text("Хранилище не подключено, не могу сохранить персону.")
        return
    if persona_text:
        redis_client.set(f"persona:{user_id}", persona_text)
        await update.message.reply_text(f"✅ Новая персона установлена:\n\n_{persona_text}_", parse_mode='Markdown')
    else:
        redis_client.delete(f"persona:{user_id}")
        await update.message.reply_text("🗑️ Персона сброшена до стандартной.")

@restricted
async def model_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("Gemini 2.5 Pro", callback_data='gemini-2.5-pro')],
        [InlineKeyboardButton("Gemini 1.5 Pro", callback_data='gemini-1.5-pro')],
        [InlineKeyboardButton("Gemini 2.5 Flash", callback_data='gemini-2.5-flash')],
        [InlineKeyboardButton("Gemini 1.5 Flash", callback_data='gemini-1.5-flash')],
        [InlineKeyboardButton("Nano Banana (Image)", callback_data='gemini-2.5-flash-image-preview')],
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
async def new_chat_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not redis_client: return
    redis_client.set(f"active_chat:{user_id}", DEFAULT_CHAT_NAME)
    redis_client.delete(f"history:{user_id}:{DEFAULT_CHAT_NAME}")
    await update.message.reply_text(f"Начат новый диалог (`{DEFAULT_CHAT_NAME}`).")

@restricted
async def save_chat_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not redis_client: return
    chat_name = " ".join(context.args).strip().replace(" ", "_")
    if not chat_name or chat_name == DEFAULT_CHAT_NAME:
        await update.message.reply_text("Пожалуйста, укажите имя для сохранения. Например: `/save_chat мой_проект`.")
        return
    active_chat = get_active_chat_name(user_id)
    current_history_json = redis_client.get(f"history:{user_id}:{active_chat}")
    if not current_history_json:
        await update.message.reply_text("Текущий диалог пуст, нечего сохранять.")
        return
    redis_client.set(f"history:{user_id}:{chat_name}", current_history_json, ex=86400 * 7)
    redis_client.sadd(f"chats:{user_id}", chat_name)
    redis_client.set(f"active_chat:{user_id}", chat_name)
    await update.message.reply_text(f"Текущий диалог сохранен как `{chat_name}` и сделан активным.")

@restricted
async def load_chat_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not redis_client: return
    chat_name = " ".join(context.args).strip()
    if not chat_name:
        await update.message.reply_text("Пожалуйста, укажите имя чата для загрузки. Например: `/load_chat мой_проект`.")
        return
    if not redis_client.sismember(f"chats:{user_id}", chat_name) and chat_name != DEFAULT_CHAT_NAME:
        await update.message.reply_text(f"Чата с именем `{chat_name}` не найдено.")
        return
    redis_client.set(f"active_chat:{user_id}", chat_name)
    await update.message.reply_text(f"Чат `{chat_name}` загружен и сделан активным.")

@restricted
async def list_chats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not redis_client: return
    active_chat = get_active_chat_name(user_id)
    all_chats = redis_client.smembers(f"chats:{user_id}")
    message = f"**Ваши диалоги:**\n\n"
    if active_chat == DEFAULT_CHAT_NAME:
        message += f"➡️ `{DEFAULT_CHAT_NAME}` (активный)\n"
    else:
        message += f"▫️ `{DEFAULT_CHAT_NAME}`\n"
    for chat in sorted(list(all_chats)):
        if chat == active_chat:
            message += f"➡️ `{chat}` (активный)\n"
        else:
            message += f"▫️ `{chat}`\n"
    await update.message.reply_text(message, parse_mode='Markdown')

@restricted
async def delete_chat_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not redis_client: return
    chat_name = " ".join(context.args).strip()
    if not chat_name or chat_name == DEFAULT_CHAT_NAME:
        await update.message.reply_text(f"Нельзя удалить чат по умолчанию. Укажите имя, например: `/delete_chat мой_проект`.")
        return
    if not redis_client.sismember(f"chats:{user_id}", chat_name):
        await update.message.reply_text(f"Чата с именем `{chat_name}` не найдено.")
        return
    redis_client.delete(f"history:{user_id}:{chat_name}")
    redis_client.srem(f"chats:{user_id}", chat_name)
    active_chat = get_active_chat_name(user_id)
    if active_chat == chat_name:
        redis_client.set(f"active_chat:{user_id}", DEFAULT_CHAT_NAME)
        await update.message.reply_text(f"Чат `{chat_name}` удален. Вы переключены на чат по умолчанию.")
    else:
        await update.message.reply_text(f"Чат `{chat_name}` удален.")

@restricted
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_message = update.message.text
    model_name = get_user_model(user_id)
    persona = get_user_persona(user_id)
    await update.message.reply_chat_action(telegram.constants.ChatAction.TYPING)
    try:
        model = genai.GenerativeModel(model_name, system_instruction=persona)
        if model_name in IMAGE_GEN_MODELS:
            image_prompt = f"Generate a high-quality, photorealistic image of: {user_message}"
            response = await model.generate_content_async(image_prompt)
            await handle_gemini_response(update, response)
            update_history(user_id, user_message, "[Запрос на генерацию изображения]")
        else:
            history = get_history(user_id)
            chat = model.start_chat(history=history)
            response_stream = await chat.send_message_async(user_message, stream=True)
            await handle_gemini_response_stream(update, response_stream, user_message)
    except Exception as e:
        logger.error(f"Ошибка при обработке текстового сообщения: {e}")
        await update.message.reply_text(f'К сожалению, произошла ошибка: {e}')

@restricted
async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    model_name = get_user_model(user_id)
    persona = get_user_persona(user_id)
    if model_name not in IMAGE_GEN_MODELS:
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
        model_gemini = genai.GenerativeModel(model_name, system_instruction=persona)
        response = await model_gemini.generate_content_async([caption, img])
        await handle_gemini_response(update, response)
    except Exception as e:
        logger.error(f"Ошибка при обработке фото: {e}")
        await update.message.reply_text(f'К сожалению, произошла ошибка при обработке фото: {e}')

@restricted
async def handle_document_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    model_name = get_user_model(user_id)
    persona = get_user_persona(user_id)
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
                img = Image.open(io.BytesIO(img_bytes))
                content_parts.append(img)
            pdf_document.close()
            await update.message.reply_text(f"Отправляю первые {num_pages} страниц PDF в Gemini на анализ...")
        elif doc.mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            document = docx.Document(file_bytes_io)
            file_text_content = "\n".join([para.text for para in document.paragraphs])
            content_parts.append(file_text_content)
        elif doc.mime_type == 'text/plain':
            file_text_content = file_bytes_io.read().decode('utf-8')
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

# --- Точка входа для постоянной работы на сервере ---
def main() -> None:
    logger.info("Создание и настройка приложения...")
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("clear", clear_history))
    application.add_handler(CommandHandler("model", model_selection))
    application.add_handler(CommandHandler("usage", usage_command))
    application.add_handler(CommandHandler("persona", persona_command))
    application.add_handler(CommandHandler("new_chat", new_chat_command))
    application.add_handler(CommandHandler("save_chat", save_chat_command))
    application.add_handler(CommandHandler("load_chat", load_chat_command))
    application.add_handler(CommandHandler("chats", list_chats_command))
    application.add_handler(CommandHandler("delete_chat", delete_chat_command))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo_message))
    supported_files_filter = filters.Document.PDF | filters.Document.DOCX | filters.Document.TXT
    application.add_handler(MessageHandler(supported_files_filter, handle_document_message))
    application.add_handler(CommandHandler("menu", menu_command))
    application.add_handler(CommandHandler("hide_menu", hide_menu_command))

    
    logger.info("Бот запущен и работает в режиме опроса...")
    application.run_polling()

if __name__ == "__main__":
    if not all([TELEGRAM_BOT_TOKEN, GEMINI_API_KEY, ALLOWED_USER_IDS_STR, redis_client]):
        logger.critical("Не все переменные окружения или подключения настроены! Бот не может запуститься.")
    else:
        main()
