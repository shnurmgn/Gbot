# -*- coding: utf-8 -*-

"""
A Telegram bot that integrates with Google's Gemini AI models.

This bot supports:
- Conversational chat with history using various Gemini text models.
- Image generation using the 'gemini-2.5-flash-image-preview' model.
- Image understanding (describing images sent by the user).
- Document analysis for PDF, DOCX, and TXT files using Pro models.
- User authentication to restrict access.
- Persistent state management using Upstash Redis for:
  - Conversation history.
  - User-selected models.
  - Custom user personas (system instructions).
  - API token usage tracking.
"""

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
import fitz  # PyMuPDF
from upstash_redis import Redis

# --- Configuration ---
# Load environment variables
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
ALLOWED_USER_IDS_STR = os.environ.get('ALLOWED_USER_IDS')
UPSTASH_REDIS_URL = os.environ.get('UPSTASH_REDIS_URL')
UPSTASH_REDIS_TOKEN = os.environ.get('UPSTASH_REDIS_TOKEN')

# Parse allowed user IDs into a list of integers
ALLOWED_USER_IDS = [int(user_id.strip()) for user_id in ALLOWED_USER_IDS_STR.split(',')] if ALLOWED_USER_IDS_STR else []

# Constants
TELEGRAM_MAX_MESSAGE_LENGTH = 4096
DOCUMENT_ANALYSIS_MODELS = ['gemini-1.5-pro', 'gemini-2.5-pro']
IMAGE_GEN_MODELS = ['gemini-2.5-flash-image-preview']
HISTORY_LIMIT = 10  # Max number of message pairs (user + model) to keep in history

# --- Redis Connection ---
redis_client = None
try:
    if UPSTASH_REDIS_URL and UPSTASH_REDIS_TOKEN:
        redis_client = Redis(url=UPSTASH_REDIS_URL, token=UPSTASH_REDIS_TOKEN)
        redis_client.ping()
        logging.info("Successfully connected to Upstash Redis.")
    else:
        logging.warning("Redis URL or Token not provided. Redis client not initialized.")
except Exception as e:
    logging.error(f"Failed to connect to Redis: {e}")
    redis_client = None

# --- Logging and Gemini API Setup ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logger.critical("GEMINI_API_KEY is not set. The bot will not be able to function.")

# --- Authorization Decorator ---
def restricted(func):
    """Decorator to restrict access to the bot to allowed users."""
    @wraps(func)
    async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        user_id = update.effective_user.id
        if user_id not in ALLOWED_USER_IDS:
            if update.message:
                await update.message.reply_text("⛔️ You do not have access to this bot.")
            elif update.callback_query:
                await update.callback_query.answer("⛔️ You do not have access to this bot.", show_alert=True)
            return
        return await func(update, context, *args, **kwargs)
    return wrapped

# --- Helper Functions ---
def update_usage_stats(user_id: int, usage_metadata):
    """Updates daily and monthly token usage stats in Redis."""
    if not redis_client or not hasattr(usage_metadata, 'total_token_count'):
        return
    try:
        total_tokens = usage_metadata.total_token_count
        today = datetime.utcnow().strftime('%Y-%m-%d')
        daily_key = f"usage:{user_id}:daily:{today}"
        redis_client.incrby(daily_key, total_tokens)
        redis_client.expire(daily_key, 86400 * 2)  # Expire in 2 days

        this_month = datetime.utcnow().strftime('%Y-%m')
        monthly_key = f"usage:{user_id}:monthly:{this_month}"
        redis_client.incrby(monthly_key, total_tokens)
        redis_client.expire(monthly_key, 86400 * 32) # Expire in 32 days
    except Exception as e:
        logger.error(f"Error updating usage statistics: {e}")

async def send_long_message(update: Update, text: str):
    """Sends a message, splitting it into multiple parts if it exceeds Telegram's limit."""
    if not text.strip():
        return
    if len(text) <= TELEGRAM_MAX_MESSAGE_LENGTH:
        await update.message.reply_text(text)
    else:
        for i in range(0, len(text), TELEGRAM_MAX_MESSAGE_LENGTH):
            await update.message.reply_text(text[i:i + TELEGRAM_MAX_MESSAGE_LENGTH])
            await asyncio.sleep(0.5) # Small delay to avoid rate limiting

async def handle_gemini_response(update: Update, response):
    """Handles non-streaming responses from the Gemini API."""
    if hasattr(response, 'usage_metadata'):
        update_usage_stats(update.effective_user.id, response.usage_metadata)
    try:
        if not response.candidates:
            block_reason = getattr(response.prompt_feedback, 'block_reason_message', 'No reason provided.')
            await update.message.reply_text(f"⚠️ Request was blocked.\nReason: {block_reason}")
            return

        candidate = response.candidates[0]
        if candidate.finish_reason.name != "STOP":
            await update.message.reply_text(f"⚠️ Content generation stopped unexpectedly. Reason: `{candidate.finish_reason.name}`", parse_mode='Markdown')
            return

        if not candidate.content.parts:
            await update.message.reply_text("The model finished but did not generate a response. Please try rephrasing your request.")
            return

        full_text = ""
        for part in candidate.content.parts:
            if hasattr(part, 'text') and part.text:
                full_text += part.text
            elif hasattr(part, 'inline_data') and part.inline_data.mime_type.startswith('image/'):
                # Send image part
                await update.message.reply_photo(photo=io.BytesIO(part.inline_data.data))
        
        # Send text part if it exists
        if full_text:
            await send_long_message(update, full_text)

    except Exception as e:
        logger.error(f"Critical error while processing Gemini response: {e}")
        await update.message.reply_text(f"A critical error occurred while processing the response: {e}")

async def handle_gemini_response_stream(update: Update, response_stream, user_message_text: str):
    """Handles streaming responses from Gemini for a better chat experience."""
    placeholder_message = None
    full_response_text = ""
    last_update_time = 0
    update_interval = 0.8  # seconds

    try:
        placeholder_message = await update.message.reply_text("...")
        last_update_time = time.time()
        
        async for chunk in response_stream:
            if hasattr(chunk, 'text') and chunk.text:
                full_response_text += chunk.text
                current_time = time.time()
                if current_time - last_update_time > update_interval:
                    try:
                        # Only edit if the message content has changed and is within length limits
                        if len(full_response_text) < TELEGRAM_MAX_MESSAGE_LENGTH - 10:
                            await placeholder_message.edit_text(full_response_text + " ✍️")
                            last_update_time = current_time
                    except telegram.error.BadRequest as e:
                        # Ignore "Message is not modified" errors
                        if "Message is not modified" not in str(e):
                            logger.warning(f"Error editing message: {e}")

        await placeholder_message.delete()
        
        if not full_response_text.strip():
            await update.message.reply_text("The model finished but did not generate a response. Please try rephrasing your request.")
            return

        await send_long_message(update, full_response_text)
        update_history(update.effective_user.id, user_message_text, full_response_text)
        
        # After streaming, the usage metadata is available on the response object
        if hasattr(response_stream, 'usage_metadata') and response_stream.usage_metadata:
            update_usage_stats(update.effective_user.id, response_stream.usage_metadata)
            
    except Exception as e:
        logger.error(f"Critical error processing streaming response from Gemini: {e}")
        if placeholder_message:
            await placeholder_message.delete()
        await update.message.reply_text(f"An error occurred during response generation: {e}")

def get_history(user_id: int) -> list:
    """Retrieves conversation history for a user from Redis."""
    if not redis_client: return []
    try:
        history_data = redis_client.get(f"history:{user_id}")
        return json.loads(history_data) if history_data else []
    except Exception:
        return []

def update_history(user_id: int, user_message_text: str, model_response_text: str):
    """Updates and prunes conversation history for a user in Redis."""
    if not redis_client: return
    history = get_history(user_id)
    history.append({'role': 'user', 'parts': [{'text': user_message_text}]})
    history.append({'role': 'model', 'parts': [{'text': model_response_text}]})
    
    # Keep history within the defined limit
    if len(history) > HISTORY_LIMIT * 2: # Each turn has 2 entries
        history = history[-(HISTORY_LIMIT * 2):]
        
    redis_client.set(f"history:{user_id}", json.dumps(history), ex=86400) # Expire in 24 hours

def get_user_model(user_id: int) -> str:
    """Gets the user's selected model from Redis, with a default fallback."""
    default_model = 'gemini-1.5-flash'
    if not redis_client: return default_model
    try:
        stored_model = redis_client.get(f"user:{user_id}:model")
        return stored_model.decode('utf-8') if stored_model else default_model
    except Exception:
        return default_model

def get_user_persona(user_id: int) -> str:
    """Gets the user's custom persona from Redis."""
    if not redis_client: return None
    persona = redis_client.get(f"persona:{user_id}")
    return persona.decode('utf-8') if persona else None

# --- Command Handlers ---
@restricted
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler for the /start command."""
    user = update.effective_user
    model_name = get_user_model(user.id)
    await update.message.reply_html(f"Hello, {user.mention_html()}!")
    await update.message.reply_text(
        f"I am a bot connected to Gemini.\nCurrent model: `{model_name}`.\n\n"
        "Use /clear to start a new conversation and clear my memory.",
        parse_mode='Markdown'
    )

@restricted
async def clear_history_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler for the /clear command."""
    user_id = update.effective_user.id
    if redis_client:
        redis_client.delete(f"history:{user_id}")
    await update.message.reply_text("Memory cleared. We can start a new conversation.")

@restricted
async def usage_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler for the /usage command to show token statistics."""
    user_id = update.effective_user.id
    if not redis_client:
        await update.message.reply_text("Storage is not connected, statistics are unavailable.")
        return
    
    today = datetime.utcnow().strftime('%Y-%m-%d')
    this_month = datetime.utcnow().strftime('%Y-%m')
    
    daily_tokens = redis_client.get(f"usage:{user_id}:daily:{today}") or 0
    monthly_tokens = redis_client.get(f"usage:{user_id}:monthly:{this_month}") or 0
    
    await update.message.reply_text(
        f"📊 **Token Usage Statistics:**\n\n"
        f"Today ({today}):\n`{int(daily_tokens):,}` tokens\n\n"
        f"This Month ({this_month}):\n`{int(monthly_tokens):,}` tokens",
        parse_mode='Markdown'
    )

@restricted
async def persona_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler for the /persona command to set a custom system prompt."""
    user_id = update.effective_user.id
    persona_text = " ".join(context.args) if context.args else None
    
    if not redis_client:
        await update.message.reply_text("Storage is not connected, I cannot save the persona.")
        return
        
    if persona_text:
        redis_client.set(f"persona:{user_id}", persona_text)
        await update.message.reply_text(f"✅ New persona has been set:\n\n_{persona_text}_", parse_mode='Markdown')
    else:
        redis_client.delete(f"persona:{user_id}")
        await update.message.reply_text("🗑️ Persona has been reset to default.")

@restricted
async def model_selection_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Displays an inline keyboard for model selection."""
    keyboard = [
        [InlineKeyboardButton("Gemini 2.5 Pro (Documents/Text)", callback_data='gemini-2.5-pro')],
        [InlineKeyboardButton("Gemini 1.5 Pro (Documents/Text)", callback_data='gemini-1.5-pro')],
        [InlineKeyboardButton("Gemini 2.5 Flash (Text)", callback_data='gemini-2.5-flash')],
        [InlineKeyboardButton("Gemini 1.5 Flash (Text)", callback_data='gemini-1.5-flash')],
        [InlineKeyboardButton("Nano Banana (Images)", callback_data='gemini-2.5-flash-image-preview')],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text('Please select a model:', reply_markup=reply_markup)

@restricted
async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles callbacks from the inline keyboard for model selection."""
    query = update.callback_query
    user_id = query.from_user.id
    await query.answer()
    selected_model = query.data
    
    if redis_client:
        redis_client.set(f"user:{user_id}:model", selected_model)
    
    await query.edit_message_text(text=f"Model changed to: `{selected_model}`. I will remember your choice.", parse_mode='Markdown')

# --- Message Handlers ---
@restricted
async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles regular text messages."""
    user_id = update.effective_user.id
    user_message = update.message.text
    model_name = get_user_model(user_id)
    persona = get_user_persona(user_id)
    
    await update.message.reply_chat_action(telegram.constants.ChatAction.TYPING)
    
    try:
        model = genai.GenerativeModel(model_name, system_instruction=persona)
        
        if model_name in IMAGE_GEN_MODELS:
            # For image generation, wrap the prompt for better results
            image_prompt = f"Generate a high-quality, photorealistic image of: {user_message}"
            response = await model.generate_content_async(image_prompt)
            await handle_gemini_response(update, response)
            update_history(user_id, user_message, "[Image generation request]")
        else:
            # For text models, use streaming chat
            history = get_history(user_id)
            chat = model.start_chat(history=history)
            response_stream = await chat.send_message_async(user_message, stream=True)
            await handle_gemini_response_stream(update, response_stream, user_message)
            
    except Exception as e:
        logger.error(f"Error handling text message: {e}")
        await update.message.reply_text(f'Unfortunately, an error occurred: {e}')

@restricted
async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles messages with photos for image understanding."""
    user_id = update.effective_user.id
    model_name = get_user_model(user_id)
    persona = get_user_persona(user_id)

    # Note: Any multi-modal model can handle photos, not just image-gen models.
    # This check could be broader, e.g., if 'flash' or 'pro' is in model_name.
    # However, keeping it specific to user's intent is fine.
    if 'flash' not in model_name and 'pro' not in model_name:
        await update.message.reply_text("To work with photos, please select a Flash or Pro model via /model.")
        return

    photo_file = await update.message.photo[-1].get_file()
    caption = update.message.caption or "Describe this image in detail."
    
    await update.message.reply_chat_action(telegram.constants.ChatAction.UPLOAD_PHOTO)
    
    try:
        photo_bytes = io.BytesIO()
        await photo_file.download_to_memory(photo_bytes)
        photo_bytes.seek(0)
        img = Image.open(photo_bytes)
        
        model_gemini = genai.GenerativeModel(model_name, system_instruction=persona)
        response = await model_gemini.generate_content_async([caption, img])
        await handle_gemini_response(update, response)
        update_history(user_id, f"[Image analysis request: {caption}]", "[Model provided image description]")
        
    except Exception as e:
        logger.error(f"Error handling photo: {e}")
        await update.message.reply_text(f'Sorry, an error occurred while processing the photo: {e}')

@restricted
async def handle_document_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles PDF, DOCX, and TXT documents."""
    user_id = update.effective_user.id
    model_name = get_user_model(user_id)
    persona = get_user_persona(user_id)
    
    if model_name not in DOCUMENT_ANALYSIS_MODELS:
        await update.message.reply_text(f"For document analysis, please select a Pro model via /model.")
        return

    doc = update.message.document
    caption = update.message.caption or "Analyze this document and provide a concise summary."
    
    await update.message.reply_text(f"Received file: {doc.file_name}.\nProcessing...")
    
    try:
        doc_file = await doc.get_file()
        file_bytes_io = io.BytesIO()
        await doc_file.download_to_memory(file_bytes_io)
        file_bytes_io.seek(0)
        
        content_parts = [caption]
        
        if doc.mime_type == 'application/pdf':
            pdf_document = fitz.open(stream=file_bytes_io.read(), filetype="pdf")
            page_limit = 25  # Limit the number of pages to avoid excessive token usage
            num_pages = min(len(pdf_document), page_limit)
            for page_num in range(num_pages):
                page = pdf_document.load_page(page_num)
                pix = page.get_pixmap()
                img_bytes = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_bytes))
                content_parts.append(img)
            pdf_document.close()
            await update.message.reply_text(f"Sending the first {num_pages} pages of the PDF to Gemini for analysis...")
            
        elif doc.mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            document = docx.Document(file_bytes_io)
            file_text_content = "\n".join([para.text for para in document.paragraphs])
            content_parts.append(file_text_content)
            
        elif doc.mime_type == 'text/plain':
            file_text_content = file_bytes_io.read().decode('utf-8')
            content_parts.append(file_text_content)
            
        else:
            await update.message.reply_text(f"Sorry, I do not support files of type {doc.mime_type} yet.")
            return

        model = genai.GenerativeModel(model_name, system_instruction=persona)
        response = await model.generate_content_async(content_parts)
        await handle_gemini_response(update, response)
        
    except Exception as e:
        logger.error(f"Error handling document: {e}")
        await update.message.reply_text(f'Sorry, an error occurred while processing the document: {e}')

# --- Main Entry Point ---
def main() -> None:
    """Initializes and runs the bot."""
    logger.info("Creating and setting up the application...")
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Register handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("clear", clear_history_command))
    application.add_handler(CommandHandler("model", model_selection_command))
    application.add_handler(CommandHandler("usage", usage_command))
    application.add_handler(CommandHandler("persona", persona_command))
    application.add_handler(CallbackQueryHandler(button_callback))
    
    # Message handlers
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo_message))
    
    # Document handler for specific MIME types
    supported_files_filter = filters.Document.PDF | filters.Document.DOCX | filters.Document.TXT
    application.add_handler(MessageHandler(supported_files_filter, handle_document_message))
    
    logger.info("Bot is running in polling mode...")
    application.run_polling()

if __name__ == "__main__":
    # Critical check for required environment variables before starting
    if not all([TELEGRAM_BOT_TOKEN, GEMINI_API_KEY, ALLOWED_USER_IDS_STR, redis_client]):
        logger.critical("Not all environment variables or connections are configured! The bot cannot start.")
    else:
        main()
