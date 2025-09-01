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
IMAGE_GEN_MODELS = ['gemini-2.5-flash-image-preview']
HISTORY_LIMIT = 10 
DEFAULT_CHAT_NAME = "default"

# --- Подключение к Upstash Redis ---
redis_client = None
try:
    redis_client = Redis(
        url=os.environ.get('UPSTASH_REDIS_URL'),
        token=os.environ.get('UPSTASH_REDIS_TOKEN'),
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
            elif update.callback_query: await update.callback_query.answer("⛔️ У вас нет доступа.", show_alert=True)
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

async def send_long_message(message: telegram.Message, text: str):
    if not text.strip(): return
    if len(text) <= TELEGRAM_MAX_MESSAGE_LENGTH:
        await message.reply_text(text, parse_mode='Markdown')
    else:
        for i in range(0, len(text), TELEGRAM_MAX_MESSAGE_LENGTH):
            await message.reply_text(text[i:i + TELEGRAM_MAX_MESSAGE_LENGTH], parse_mode='Markdown')
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
            await send_long_message(update.message, full_text)
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
        
        await placeholder_message.delete()
        
        if not full_response_text.strip():
             await update.message.reply_text("Модель завершила работу, но не сгенерировала ответ. Попробуйте переформулировать ваш запрос.")
             return

        await send_long_message(update.message, full_response_text)
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

async def get_main_menu_text_and_keyboard(user_id: int):
    """Собирает текст и клавиатуру для главного меню."""
    model_name = get_user_model(user_id)
    active_chat = get_active_chat_name(user_id)
    text = (
        f"🤖 **Главное меню**\n\n"
        f"Текущая модель: `{model_name}`\n"
        f"Текущий чат: `{active_chat}`\n\n"
        f"Выберите действие:"
    )
    # --- ИЗМЕНЕНИЕ: Добавляем кнопку помощи ---
    keyboard = [
        [
            InlineKeyboardButton("🤖 Выбрать модель", callback_data="menu:model"),
            InlineKeyboardButton("👤 Персона", callback_data="menu:persona")
        ],
        [
            InlineKeyboardButton("💬 Управление чатами", callback_data="menu:open_chats_submenu")
        ],
        [
            InlineKeyboardButton("🗑️ Очистить чат", callback_data="menu:clear"),
            InlineKeyboardButton("📈 Статистика", callback_data="menu:usage")
        ],
        [
            InlineKeyboardButton("❓ Что умеет бот?", callback_data="menu:help")
        ]
    ]
    return text, InlineKeyboardMarkup(keyboard)

async def get_chats_submenu_text_and_keyboard():
    """Собирает текст и клавиатуру для ПОДМЕНЮ чатов."""
    text = "🗂️ **Управление чатами**"
    keyboard = [
        [InlineKeyboardButton("📖 Сохраненные чаты", callback_data="chats:list")],
        [InlineKeyboardButton("📥 Сохранить текущий чат", callback_data="chats:save")],
        [InlineKeyboardButton("➕ Новый чат", callback_data="chats:new")],
        [InlineKeyboardButton("⬅️ Назад в меню", callback_data="menu:main")]
    ]
    return text, InlineKeyboardMarkup(keyboard)

# --- НОВАЯ ФУНКЦИЯ ---
@restricted
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE, from_callback: bool = False):
    """Отправляет сообщение со справкой о возможностях бота."""
    help_text = (
        "🤖 **Привет! Вот что я умею:**\n\n"
        "Я ваш персональный ассистент на базе Google Gemini. Я могу общаться с вами, помнить контекст, генерировать и анализировать контент.\n\n"
        "**Основные возможности:**\n\n"
        "💬 **Диалог с памятью**\n"
        "Просто общайтесь со мной. Я помню последние 10 сообщений, чтобы вы могли задавать уточняющие вопросы.\n\n"
        "🖼️ **Работа с изображениями**\n"
        "• **Генерация:** Выберите модель `Nano Banana` и попросите нарисовать что-нибудь (например, `нарисуй кота-астронавта`).\n"
        "• **Анализ:** Отправьте мне фото с вопросом в подписи (например, `что на этой картинке?`).\n\n"
        "📄 **Анализ документов**\n"
        "Отправьте мне файл (`.pdf`, `.docx`, `.txt`) с задачей в подписи (например, `сделай краткую выжимку`). Для этого лучше всего подходят модели `Pro`.\n\n"
        "**Команды управления:**\n"
        "• `/menu` - Показать главное меню с кнопками.\n"
        "• `/persona <текст>` - Установить мне личность. Пустая команда `/persona` сбрасывает ее.\n"
        "• `/usage` - Посмотреть статистику использования токенов.\n"
        "• `/clear` - Очистить историю текущего чата.\n\n"
        "**Управление чатами:**\n"
        "• `/new_chat` - Начать новый диалог.\n"
        "• `/save_chat <имя>` - Сохранить текущий диалог.\n"
        "• `/load_chat <имя>` - Загрузить сохраненный диалог.\n"
        "• `/chats` - Показать список ваших диалогов.\n"
        "• `/delete_chat <имя>` - Удалить диалог."
    )
    
    if from_callback:
        keyboard = [[InlineKeyboardButton("⬅️ Назад в меню", callback_data='menu:main')]]
        await update.callback_query.edit_message_text(help_text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='Markdown')
    else:
        await update.message.reply_text(help_text, parse_mode='Markdown')

@restricted
async def main_menu_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    menu_text, reply_markup = await get_main_menu_text_and_keyboard(user_id)
    target_message = update.callback_query.message if update.callback_query else update.message
    try:
        await target_message.edit_text(menu_text, reply_markup=reply_markup, parse_mode='Markdown')
    except (AttributeError, telegram.error.BadRequest):
        if update.message:
            try: await update.message.delete()
            except: pass
        await context.bot.send_message(chat_id=user_id, text=menu_text, reply_markup=reply_markup, parse_mode='Markdown')

async def clear_history_logic(update: Update):
    user_id = update.effective_user.id
    active_chat = get_active_chat_name(user_id)
    if redis_client: redis_client.delete(f"history:{user_id}:{active_chat}")
    return f"Память текущего чата (`{active_chat}`) очищена."

@restricted
async def clear_history_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    response_text = await clear_history_logic(update)
    await update.message.reply_text(response_text, parse_mode='Markdown')

@restricted
async def usage_command(update: Update, context: ContextTypes.DEFAULT_TYPE, from_callback: bool = False):
    user_id = update.effective_user.id
    if not redis_client:
        await update.message.reply_text("Хранилище не подключено, статистика недоступна.")
        return
    today = datetime.utcnow().strftime('%Y-%m-%d')
    this_month = datetime.utcnow().strftime('%Y-%m')
    daily_tokens = redis_client.get(f"usage:{user_id}:daily:{today}") or 0
    monthly_tokens = redis_client.get(f"usage:{user_id}:monthly:{this_month}") or 0
    text = (
        f"📊 **Статистика использования токенов:**\n\n"
        f"Сегодня ({today}):\n`{int(daily_tokens):,}` токенов\n\n"
        f"В этом месяце ({this_month}):\n`{int(monthly_tokens):,}` токенов"
    )
    if from_callback:
        keyboard = [[InlineKeyboardButton("⬅️ Назад в меню", callback_data='menu:main')]]
        await update.callback_query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='Markdown')
    else:
        await update.message.reply_text(text, parse_mode='Markdown')

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
async def model_selection_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("Gemini 2.5 Pro", callback_data='select_model:gemini-2.5-pro')],
        [InlineKeyboardButton("Gemini 1.5 Pro", callback_data='select_model:gemini-1.5-pro')],
        [InlineKeyboardButton("Gemini 2.5 Flash", callback_data='select_model:gemini-2.5-flash')],
        [InlineKeyboardButton("Gemini 1.5 Flash", callback_data='select_model:gemini-1.5-flash')],
        [InlineKeyboardButton("Nano Banana (Image)", callback_data='select_model:gemini-2.5-flash-image-preview')],
        [InlineKeyboardButton("⬅️ Назад в меню", callback_data='menu:main')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.edit_message_text('Выберите модель:', reply_markup=reply_markup)

@restricted
async def new_chat_command(update: Update, context: ContextTypes.DEFAULT_TYPE, from_callback: bool = False):
    user_id = update.effective_user.id
    if not redis_client: return
    redis_client.set(f"active_chat:{user_id}", DEFAULT_CHAT_NAME)
    redis_client.delete(f"history:{user_id}:{DEFAULT_CHAT_NAME}")
    response_text = f"Начат новый диалог (`{DEFAULT_CHAT_NAME}`)."
    target_message = update.callback_query.message if from_callback else update.message
    await target_message.reply_text(response_text, parse_mode='Markdown')

@restricted
async def save_chat_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not redis_client: return
    chat_name = "_".join(context.args).strip()
    if not chat_name or chat_name == DEFAULT_CHAT_NAME:
        await update.message.reply_text("Пожалуйста, укажите имя для сохранения. Например: `/save_chat мой проект`.")
        return
    active_chat = get_active_chat_name(user_id)
    current_history_json = redis_client.get(f"history:{user_id}:{active_chat}")
    if not current_history_json:
        await update.message.reply_text("Текущий диалог пуст, нечего сохранять.")
        return
    redis_client.set(f"history:{user_id}:{chat_name}", current_history_json, ex=86400 * 7)
    redis_client.sadd(f"chats:{user_id}", chat_name)
    redis_client.set(f"active_chat:{user_id}", chat_name)
    await update.message.reply_text(f"Текущий диалог сохранен как `{chat_name}` и сделан активным.", parse_mode='Markdown')

@restricted
async def load_chat_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not redis_client: return
    chat_name = "_".join(context.args).strip()
    if not chat_name:
        await update.message.reply_text("Пожалуйста, укажите имя чата для загрузки. Например: `/load_chat мой_проект`.")
        return
    if not redis_client.sismember(f"chats:{user_id}", chat_name) and chat_name != DEFAULT_CHAT_NAME:
        await update.message.reply_text(f"Чата с именем `{chat_name}` не найдено.", parse_mode='Markdown')
        return
    redis_client.set(f"active_chat:{user_id}", chat_name)
    await update.message.reply_text(f"Чат `{chat_name}` загружен и сделан активным.", parse_mode='Markdown')

@restricted
async def list_chats_command(update: Update, context: ContextTypes.DEFAULT_TYPE, from_callback: bool = False):
    user_id = update.effective_user.id
    if not redis_client: return
    active_chat = get_active_chat_name(user_id)
    all_chats = redis_client.smembers(f"chats:{user_id}")
    message = f"**Ваши диалоги:**\n\n"
    if active_chat == DEFAULT_CHAT_NAME:
        message += f"➡️ `{DEFAULT_CHAT_NAME}` (активный)\n"
    else:
        message += f"▫️ `{DEFAULT_CHAT_NAME}` (`/new_chat`)\n"
    for chat in sorted(list(all_chats)):
        if chat == active_chat:
            message += f"➡️ `{chat}` (активный)\n"
        else:
            message += f"▫️ `{chat}` (`/load_chat {chat}`)\n"
    
    keyboard = [[InlineKeyboardButton("⬅️ Назад в меню", callback_data='menu:main')]]
    
    if from_callback:
        await update.callback_query.edit_message_text(message, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='Markdown')
    else:
        await update.message.reply_text(message, parse_mode='Markdown')

@restricted
async def delete_chat_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not redis_client: return
    chat_name = "_".join(context.args).strip()
    if not chat_name or chat_name == DEFAULT_CHAT_NAME:
        await update.message.reply_text(f"Нельзя удалить чат по умолчанию. Укажите имя, например: `/delete_chat мой_проект`.")
        return
    if not redis_client.sismember(f"chats:{user_id}", chat_name):
        await update.message.reply_text(f"Чата с именем `{chat_name}` не найдено.", parse_mode='Markdown')
        return
    redis_client.delete(f"history:{user_id}:{chat_name}")
    redis_client.srem(f"chats:{user_id}", chat_name)
    active_chat = get_active_chat_name(user_id)
    if active_chat == chat_name:
        redis_client.set(f"active_chat:{user_id}", DEFAULT_CHAT_NAME)
        await update.message.reply_text(f"Чат `{chat_name}` удален. Вы переключены на чат по умолчанию.", parse_mode='Markdown')
    else:
        await update.message.reply_text(f"Чат `{chat_name}` удален.", parse_mode='Markdown')

@restricted
async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    command, *payload = query.data.split(':', 1)
    payload = payload[0] if payload else None
    
    if command == "menu":
        if payload == "model":
            await model_selection_menu(update, context)
        elif payload == "persona":
            await query.message.reply_text("Отправьте команду:\n`/persona <текст>` для установки,\n`/persona` без текста для сброса.", parse_mode='Markdown')
        elif payload == "open_chats_submenu":
            submenu_text, reply_markup = await get_chats_submenu_text_and_keyboard()
            await query.edit_message_text(submenu_text, reply_markup=reply_markup, parse_mode='Markdown')
        elif payload == "clear":
            response_text = await clear_history_logic(update)
            # Отправляем подтверждение как новое сообщение, чтобы оно не исчезло
            await query.message.reply_text(response_text, parse_mode='Markdown')
            await menu_command(update, context)
        elif payload == "usage":
            await usage_command(update, context, from_callback=True)
        elif payload == "help":
            await help_command(update, context, from_callback=True)
        elif payload == "main":
            await menu_command(update, context)

    elif command == "chats":
        if payload == "list":
            await list_chats_command(update, context, from_callback=True)
        elif payload == "save":
            await query.message.reply_text("Чтобы сохранить текущий чат, отправьте команду:\n`/save_chat <имя_чата>`\nПробелы будут заменены на `_`.", parse_mode='Markdown')
        elif payload == "new":
            await new_chat_command(update, context, from_callback=True)
            await menu_command(update, context)
            
    elif command == "select_model":
        user_id = query.from_user.id
        if redis_client: redis_client.set(f"user:{user_id}:model", payload)
        menu_text, reply_markup = await get_main_menu_text_and_keyboard(user_id)
        try:
            await query.edit_message_text(
                f"✅ Модель изменена на `{payload}`.\n\n" + menu_text,
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
        except telegram.error.BadRequest: pass

@restricted
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # ... (код без изменений)

@restricted
async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # ... (код без изменений)

@restricted
async def handle_document_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # ... (код без изменений)

# --- Точка входа для сервера ---
def main() -> None:
    logger.info("Создание и настройка приложения...")
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    application.add_handler(CommandHandler(["start", "menu"], main_menu_command))
    application.add_handler(CommandHandler("help", help_command)) # <-- НОВАЯ КОМАНДА
    application.add_handler(CommandHandler("clear", clear_history_command))
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
    
    logger.info("Бот запущен и работает в режиме опроса...")
    application.run_polling()

if __name__ == "__main__":
    if not all([TELEGRAM_BOT_TOKEN, GEMINI_API_KEY, ALLOWED_USER_IDS_STR, redis_client]):
        logger.critical("Не все переменные окружения или подключения настроены! Бот не может запуститься.")
    else:
        main()
