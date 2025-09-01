import logging
import redis
import asyncio
import telegram
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)
import fitz  # PyMuPDF
import google.generativeai as genai

# ---------------- CONFIG ---------------- #
TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
DEFAULT_MODEL = "gemini-2.5-pro"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- REDIS ---------------- #
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)


def _decode(val, default=None):
    if not val:
        return default
    return val.decode("utf-8") if isinstance(val, (bytes, bytearray)) else val


# ---------------- GEMINI INIT ---------------- #
genai.configure(api_key=GEMINI_API_KEY)


def get_user_model(user_id):
    stored_model = redis_client.get(f"user:{user_id}:model")
    return _decode(stored_model, DEFAULT_MODEL)


def set_user_model(user_id, model_name):
    redis_client.set(f"user:{user_id}:model", model_name)


# ---------------- HANDLERS ---------------- #
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("📋 Главное меню", callback_data="main_menu")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "👋 Привет! Я бот на базе Gemini.\n\n"
        "Используй кнопки ниже или команды, чтобы работать со мной.",
        reply_markup=reply_markup,
    )


async def main_menu_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("💬 Новый диалог", callback_data="new_chat")],
        [InlineKeyboardButton("📂 Сохранить чат", callback_data="save_chat")],
        [InlineKeyboardButton("🗂 Мои чаты", callback_data="list_chats")],
        [InlineKeyboardButton("⚙️ Настройки", callback_data="settings")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    if update.callback_query:
        await update.callback_query.message.edit_text(
            "📋 Главное меню:", reply_markup=reply_markup
        )
    else:
        await update.message.reply_text("📋 Главное меню:", reply_markup=reply_markup)


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if query.data == "main_menu":
        await main_menu_command(update, context)
    elif query.data == "new_chat":
        await query.message.reply_text("🆕 Новый чат создан!")
    elif query.data == "save_chat":
        await query.message.reply_text("💾 Чат сохранён!")
    elif query.data == "list_chats":
        await query.message.reply_text("📂 Вот список сохранённых чатов:")
    elif query.data == "settings":
        await query.message.reply_text("⚙️ Настройки пока недоступны.")


# ---------------- CHAT ---------------- #
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text

    await update.message.chat.send_action(telegram.constants.ChatAction.TYPING)

    model_name = get_user_model(user_id)
    model = genai.GenerativeModel(model_name)

    try:
        response = model.generate_content(text)
        candidate = response.candidates[0]

        if not candidate.finish_reason or candidate.finish_reason.name != "STOP":
            await update.message.reply_text("⚠️ Ответ модели был прерван.")
            return

        reply_text = candidate.content.parts[0].text
        await update.message.reply_text(reply_text)

    except Exception as e:
        logger.error(f"Ошибка Gemini: {e}")
        await update.message.reply_text("❌ Ошибка при запросе к модели.")


# ---------------- FILE HANDLING ---------------- #
async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    document = update.message.document
    if not document:
        return

    file = await document.get_file()
    file_path = f"/tmp/{document.file_name}"
    await file.download_to_drive(file_path)

    if document.file_name.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
        preview = text[:1000] + "..." if len(text) > 1000 else text
        await update.message.reply_text(f"📄 Текст из PDF:\n\n{preview}")
    else:
        await update.message.reply_text("❌ Поддерживаются только PDF-файлы.")


def extract_text_from_pdf(pdf_path, max_pages=25):
    text = ""
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            text += page.get_text()
    return text


# ---------------- MAIN ---------------- #
def main():
    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("menu", main_menu_command))
    application.add_handler(CallbackQueryHandler(button_callback))

    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )
    application.add_handler(MessageHandler(filters.Document.ALL, handle_file))

    logger.info("Бот запущен...")
    application.run_polling()


if __name__ == "__main__":
    main()
