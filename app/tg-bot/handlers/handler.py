import logging
import uuid
from io import BytesIO

import requests
from aiogram import Bot, Router, types
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup


BACKEND_HOST = "backend"
BACKEND_PORT = "5000"

# Configure logging with DEBUG level
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TG-BOT-HANDLER")

router = Router()

# Define states for the process
class UserStates(StatesGroup):
    waiting_for_photo = State()
    waiting_for_rating = State()
    waiting_for_transcription = State()

# Store request_id and rating for transcription
user_requests = {}

@router.message(Command('start'))
async def start(message: Message, state: FSMContext):
    await state.set_state(UserStates.waiting_for_photo)
    await message.answer('Пожалуйста, загрузите изображение с текстом.')

@router.message(Command('help'))
async def help(message: Message, state: FSMContext):
    await state.set_state(UserStates.waiting_for_photo)
    await message.answer('Этот бот поможет расшифровать написанное от руки. Пожалуйста, загрузите изображение.')

@router.message(Command("clear"))
async def all_clear(message: Message, bot: Bot, state: FSMContext):
    try:
        for i in range(message.message_id, 0, -1):
            await bot.delete_message(message.from_user.id, i)
    except TelegramBadRequest as ex:
        if ex.message == "Bad Request: message to delete not found":
            logger.info("Все сообщения удалены!")
    await state.set_state(UserStates.waiting_for_photo)

def get_rating_keyboard():
    """Create inline keyboard with rating buttons"""
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="1 ⭐", callback_data="rate_1"),
            InlineKeyboardButton(text="2 ⭐", callback_data="rate_2"),
            InlineKeyboardButton(text="3 ⭐", callback_data="rate_3"),
        ],
        [
            InlineKeyboardButton(text="4 ⭐", callback_data="rate_4"),
            InlineKeyboardButton(text="5 ⭐", callback_data="rate_5"),
        ],
        [
            InlineKeyboardButton(text="Пропустить", callback_data="rate_skip")
        ]
    ])
    return keyboard

@router.message(UserStates.waiting_for_photo)
async def photo_handler(message: types.Message, bot: Bot, state: FSMContext):
    if message.content_type == types.ContentType.PHOTO:
        logger.debug("Processing new photo")
        
        file = await bot.get_file(message.photo[-1].file_id)
        file_path = file.file_path
        img_data: BytesIO = await bot.download_file(file_path)

        await bot.send_message(message.from_user.id, "Фото загружено, идет обработка!")

        url = f'http://{BACKEND_HOST}:{BACKEND_PORT}/predict'
        request_id = str(uuid.uuid4())
        files = {'file': img_data}
        data = {'user_id': str(message.from_user.id), 'request_id': request_id}
        response = requests.post(url, files=files, data=data)
        
        if response.status_code == 200:
            answer = response.json()
            # Store request_id for this user
            user_requests[message.from_user.id] = request_id
            
            # Send prediction result with rating keyboard
            await bot.send_message(
                message.from_user.id,
                f"Ответ модели: {answer['prediction']}, уверенность {answer['confidence']*100:.2f}%"
            )
            await bot.send_message(
                message.from_user.id,
                "Оцените качество распознавания:",
                reply_markup=get_rating_keyboard()
            )
            # Set state to waiting for rating
            await state.set_state(UserStates.waiting_for_rating)
        else:
            await bot.send_message(message.from_user.id, "Что-то пошло не так")
            await state.set_state(UserStates.waiting_for_photo)

@router.callback_query(lambda c: c.data.startswith('rate_'))
async def process_rating(callback_query: types.CallbackQuery, bot: Bot, state: FSMContext):
    current_state = await state.get_state()
    logger.debug(f"Processing rating. Current state: {current_state}")
    
    if current_state != UserStates.waiting_for_rating:
        logger.debug("Not in rating state, ignoring callback")
        return

    user_id = callback_query.from_user.id
    rating_data = callback_query.data
    logger.debug(f"Rating data: {rating_data}")
    
    if rating_data == "rate_skip":
        logger.debug("Rating skipped")
        await callback_query.message.edit_text(
            callback_query.message.text + "\n\nОценка пропущена"
        )
        await callback_query.answer()
        await state.set_state(UserStates.waiting_for_photo)
        return
    
    rating = int(rating_data.split('_')[1])
    request_id = user_requests.get(user_id)
    logger.debug(f"Rating: {rating}, Request ID: {request_id}")
    
    if request_id:
        # Send rating to backend
        url = f'http://{BACKEND_HOST}:{BACKEND_PORT}/rate'
        data = {
            'request_id': request_id,
            'rating': rating
        }
        response = requests.post(url, json=data)
        
        if response.status_code == 200:
            if rating <= 3:
                logger.debug("Setting transcription state")
                await state.set_state(UserStates.waiting_for_transcription)
                user_requests[user_id] = {
                    'request_id': request_id,
                    'rating': rating
                }
                await callback_query.message.edit_text(
                    callback_query.message.text + f"\n\nСпасибо за оценку: {rating} ⭐\n"
                    "Пожалуйста, напишите правильный текст, который должен был быть распознан:"
                )
            else:
                logger.debug("High rating, returning to photo state")
                await callback_query.message.edit_text(
                    callback_query.message.text + f"\n\nСпасибо за оценку: {rating} ⭐"
                )
                user_requests.pop(user_id, None)
                await state.set_state(UserStates.waiting_for_photo)
        else:
            logger.error(f"Failed to save rating. Status code: {response.status_code}")
            await callback_query.message.edit_text(
                callback_query.message.text + "\n\nНе удалось сохранить оценку"
            )
            user_requests.pop(user_id, None)
            await state.set_state(UserStates.waiting_for_photo)
    
    await callback_query.answer()

@router.message(UserStates.waiting_for_transcription)
async def process_transcription(message: Message, bot: Bot, state: FSMContext):
    user_id = message.from_user.id
    current_state = await state.get_state()
    logger.debug(f"Processing transcription. Current state: {current_state}")
    
    user_data = user_requests.get(user_id)
    logger.debug(f"User data: {user_data}")
    
    if user_data:
        request_id = user_data['request_id']
        transcription = message.text
        
        # Send transcription to backend
        url = f'http://{BACKEND_HOST}:{BACKEND_PORT}/transcribe'
        data = {
            'request_id': request_id,
            'transcription': transcription
        }
        response = requests.post(url, json=data)
        
        if response.status_code == 200:
            logger.debug("Transcription saved successfully")
            await message.answer("Спасибо за предоставление правильного текста!")
        else:
            logger.error(f"Failed to save transcription. Status code: {response.status_code}")
            await message.answer("Не удалось сохранить правильный текст")
        
        # Clean up
        user_requests.pop(user_id, None)
        await state.set_state(UserStates.waiting_for_photo)
        
        # Send a new message to indicate the process is complete
        await message.answer("Теперь вы можете отправить новое изображение для распознавания.")
    else:
        logger.error("No user data found for transcription")
        await message.answer("Произошла ошибка. Пожалуйста, попробуйте еще раз.")
        await state.set_state(UserStates.waiting_for_photo)
