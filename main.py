import os
import logging
import asyncio
from datetime import datetime
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.filters import CommandStart, Command
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
from aiohttp import web
from dotenv import load_dotenv
from supabase import create_client, Client

# Импорт оригинальной функции из твоего analyzer.py
from analyzer import ANALYZER_BACKEND, analyze_face

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not BOT_TOKEN:
    logger.critical("BOT_TOKEN missing in .env configuration!")
    raise ValueError("BOT_TOKEN not found in .env")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# Инициализация Supabase с проверкой валидности URL
supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        # Убираем случайные пробелы и кавычки, если они пролезли из env
        clean_url = SUPABASE_URL.strip().replace('"', '').replace("'", "")
        clean_key = SUPABASE_KEY.strip().replace('"', '').replace("'", "")
        
        supabase = create_client(clean_url, clean_key)
        logger.info("Supabase client initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
else:
    logger.warning("Supabase credentials missing. History feature disabled.")

# Состояния FSM для контроля шагов пользователя
class AnalyzerStates(StatesGroup):
    waiting_for_photo = State()


from aiogram.utils.keyboard import ReplyKeyboardBuilder
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

# --- КЛАВИАТУРЫ ---

def get_main_keyboard() -> ReplyKeyboardMarkup:
    builder = ReplyKeyboardBuilder()
    
    # Создаем 4 главные текстовые кнопки
    builder.button(text="📊 Анализ лица")
    builder.button(text="🏆 Таблица моггеров")
    builder.button(text="💎 Премиум")
    builder.button(text="📜 История")
    
    # Выстраиваем их сеткой 2х2
    builder.adjust(2)
    
    return builder.as_markup(
        resize_keyboard=True,
        input_field_placeholder="Выберите нужный раздел..."
    )

# --- ХЭНДЛЕРЫ ---

@dp.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext):
    await state.clear()
    logger.info(f"User {message.from_user.id} started the bot.")
    
    welcome_text = (
        "🤖 Добро пожаловать в Face Analyzer AI!\n\n"
        "Я помогу провести глубокий компьютерный анализ пропорций и симметрии твоего лица при помощи MediaPipe и ИИ.\n\n"
        "Выберите интересующий пункт меню на клавиатуре ниже"
    )
    # Отправляем меню с нашей новой Reply-клавиатурой (без parse_mode, чистый текст)
    await message.answer(welcome_text, reply_markup=get_main_keyboard())


@dp.message(F.text == "📊 Анализ лица")
async def process_rate_photo(message: Message, state: FSMContext):
    await state.clear()
    await state.set_state(AnalyzerStates.waiting_for_photo)
    
    instruction_text = (
        "📸 Отправьте фотографию лица\n\n"
        "⚠️ Требования к снимку для точной работы ИИ:\n"
        "• Лицо смотрит строго прямо (анфас)\n"
        "• Хорошее, равномерное освещение\n"
        "• Без медицинских масок и солнцезащитных очков\n"
        "• На фотографии должен быть строго один человек\n\n"
        "Просто пришлите фото в чат..."
    )
    await message.answer(instruction_text)


@dp.message(F.text == "🏆 Таблица моггеров")
async def process_leaderboard(message: Message, state: FSMContext):
    await state.clear()
    
    # Твоя будущая таблица лидеров, пока что красивый статичный топ
    leaderboard_text = (
        "🏆 ТОП-5 МОГГЕРОВ БОТА\n\n"
        "1. 👑 Султан — 8.1/10 (Симметрия: 8.9)\n"
        "2. ⚡ Шахзод — 7.8/10 (Симметрия: 8.4)\n"
        "3. 🧊 Даврон — 7.5/10 (Симметрия: 7.9)\n"
        "4. 👾 Алекс — 7.2/10 (Симметрия: 7.5)\n"
        "5. ☄️ Рустам — 7.0/10 (Симметрия: 7.2)\n\n"
        "Хотите попасть в таблицу? Пройдите анализ лица с идеальным освещением!"
    )
    await message.answer(leaderboard_text)


@dp.message(F.text == "💎 Премиум")
async def process_premium(message: Message, state: FSMContext):
    await state.clear()
    
    premium_text = (
        "💎 Face Analyzer Premium\n\n"
        "Откройте расширенные возможности ИИ аналитики:\n"
        "• Отсутствие лимитов на загрузку фото.\n"
        "• Подробный расчет Hunter Eyes и углов челюсти.\n"
        "• Персональный ИИ-стилист по подбору причесок.\n\n"
        "Функция Premium интеграции находится в разработке (скоро через Telegram Stars)."
    )
    await message.answer(premium_text)


@dp.message(F.text == "📜 История")
async def process_history(message: Message, state: FSMContext):
    await state.clear()
    user_id = message.from_user.id
    
    if not supabase:
        await message.answer("⚠️ Система хранения истории отключена или недоступна.")
        return

    try:
        response = (
            supabase.table("photo_ratings")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(5)
            .execute()
        )
        
        if not response.data:
            await message.answer(
                "📜 Ваша история пуста\n\nВы еще не сканировали фотографии в нашем боте."
            )
            return
            
        history_text = "📜 Ваши последние 5 сканирований:\n\n"
        for idx, row in enumerate(response.data, 1):
            raw_date = row.get('created_at', '-----')
            formatted_date = raw_date[:10] if len(raw_date) >= 10 else raw_date
            score = row.get('score', 0.0)
            history_text += f"{idx}. Оценка: ⭐ {score}/10 | Дата: {formatted_date}\n"
            
        await message.answer(history_text)
        
    except Exception as e:
        logger.error(f"Error querying history from Supabase for user {user_id}: {e}")
        await message.answer("❌ Произошла техническая ошибка при чтении истории.")

# --- ОБРАБОТКА ФОТО (FSM СЦЕНАРИЙ) ---

@dp.message(AnalyzerStates.waiting_for_photo, F.photo)
async def handle_photo_analysis(message: Message, state: FSMContext):
    user_id = message.from_user.id
    username = message.from_user.username or "unknown"
    
    waiting_msg = await message.answer("🔄 **Инициализация процесса...**\n1. Проверяем наличие лица.\n2. Находим ключевые точки лица.\n3. Анализируем пропорции.\n\n*Пожалуйста, подождите.*")
    
    photo = message.photo[-1]
    temp_image_path = f"scan_{user_id}_{photo.file_id[:10]}.jpg"
    
    try:
        logger.info(f"Downloading image from user {user_id}...")
        await bot.download(photo, destination=temp_image_path)
        
        logger.info(f"Executing analytical core for file {temp_image_path}...")
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, analyze_face, temp_image_path)
        
        if not result or not isinstance(result, dict):
            raise ValueError("Invalid data format returned from backend core.")

        score = result.get("score", 0.0)
        report = result.get("report", "⚠️ Отчет пуст.")

        if score > 0 and supabase:
            try:
                supabase.table("photo_ratings").insert({
                    "user_id": user_id,
                    "username": username,
                    "score": score,
                    "report": report
                }).execute()
                logger.info(f"Analysis results saved to Supabase for user {user_id}")
            except Exception as db_err:
                logger.error(f"Failed to record statistics in Supabase: {db_err}")

        await bot.delete_message(chat_id=message.chat.id, message_id=waiting_msg.message_id)
        
        await message.reply(report)
        await state.clear()

    except Exception as e:
        logger.error(f"Critical error during photo processing chain: {e}", exc_info=True)
        await bot.delete_message(chat_id=message.chat.id, message_id=waiting_msg.message_id)
        await message.answer(
            f"❌ **Сбой внутренней обработки данных**\n\nПроизошла непредвиденная ошибка на сервере.\nДетали: `{str(e)}`",
            parse_mode="Markdown"
        )
    finally:
        if os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
                logger.info(f"Temporary file {temp_image_path} cleared from host storage.")
            except Exception as ce:
                logger.error(f"Failed to clear temp file {temp_image_path}: {ce}")

@dp.message(AnalyzerStates.waiting_for_photo)
async def handle_invalid_input_type(message: Message):
    await message.reply(
        "⚠️ **Некорректный формат данных**\n\nСистема ожичает прямую передачу графического файла (изображения/фотографии).\nПожалуйста, пришлите снимок лица или нажмите кнопку возврата ниже.",
        reply_markup=get_back_btn()
    )

# --- МОНИТОРИНГ И СИСТЕМНЫЕ КЛИЕНТЫ ---

@dp.message(Command("health"))
async def system_health_check(message: Message):
    db_status = "ONLINE" if supabase else "OFFLINE"
    uptime_text = (
        "⚙️ **System Diagnostics Status:**\n\n"
        f"• Gateway API: `Aiogram 3.x Long-Polling` \n"
        f"• Database Node: `{db_status}`\n"
        f"• Analyzer Backend: `{ANALYZER_BACKEND}`\n"
        f"• Host Machine Node: `Render Cloud Environment` \n"
        f"• Request Timestamp: `{datetime.utcnow().isoformat()}`"
    )
    await message.answer(uptime_text, parse_mode="Markdown")

async def healthcheck(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok", "analyzer_backend": ANALYZER_BACKEND})
    return web.json_response({"status": "ok"})


async def start_health_server() -> web.AppRunner | None:
    port = os.getenv("PORT")

    if not port:
        logger.info("PORT is not set; health web server is disabled.")
        return None

    app = web.Application()
    app.router.add_get("/", healthcheck)
    app.router.add_get("/health", healthcheck)

    runner = web.AppRunner(app)
    await runner.setup()

    site = web.TCPSite(runner, host="0.0.0.0", port=int(port))
    await site.start()

    logger.info("Health web server is listening on port %s.", port)
    return runner


async def main():
    logger.info("Initializing polling engine with analyzer backend: %s", ANALYZER_BACKEND)
    logger.info("Initializing polling engine...")
    health_runner = await start_health_server()

    try:
        await dp.start_polling(bot)
    finally:
        if health_runner:
            await health_runner.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot application context terminated.")
