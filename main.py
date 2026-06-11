import os
import sys

# Находим, где лежат установленные библиотеки Python
site_packages = [p for p in sys.path if 'site-packages' in p]
if site_packages:
    # Прописываем путь к бинарникам panda3d, где лежит заветный libGLESv2
    panda_path = os.path.join(site_packages[0], 'panda3d')
    if os.path.exists(panda_path):
        os.environ['LD_LIBRARY_PATH'] = f"{panda_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"

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

# --- КЛАВИАТУРЫ ---

def get_main_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="📸 Оценить фото", callback_data="menu_rate")],
            [InlineKeyboardButton(text="📊 История", callback_data="menu_history"), InlineKeyboardButton(text="❓ Как работает", callback_data="menu_how_it_works")],
            [InlineKeyboardButton(text="⭐ Premium", callback_data="menu_premium")]
        ]
    )

def get_back_btn(target: str = "main_menu") -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text="⬅️ Назад в меню", callback_data=target)]]
    )

# --- ХЭНДЛЕРЫ ---

@dp.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext):
    await state.clear()
    logger.info(f"User {message.from_user.id} started the bot.")
    
    welcome_text = (
        "🤖 **Добро пожаловать в Face Analyzer AI!**\n\n"
        "Я помогу провести глубокий компьютерный анализ пропорций и симметрии твоего лица при помощи MediaPipe и ИИ.\n\n"
        "Выберите интересующий пункт меню ниже:"
    )
    await message.answer(welcome_text, reply_markup=get_main_menu(), parse_mode="Markdown")

@dp.callback_query(F.data == "main_menu")
async def back_to_main_menu(callback: CallbackQuery, state: FSMContext):
    await state.clear()
    await callback.answer()
    welcome_text = (
        "🤖 **Главное меню Face Analyzer AI**\n\n"
        "Выберите действие:"
    )
    try:
        await callback.message.edit_text(welcome_text, reply_markup=get_main_menu(), parse_mode="Markdown")
    except Exception:
        await callback.message.answer(welcome_text, reply_markup=get_main_menu(), parse_mode="Markdown")

@dp.callback_query(F.data == "menu_how_it_works")
async def process_how_it_works(callback: CallbackQuery):
    await callback.answer()
    info_text = (
        "❓ **Как устроен анализ лица?**\n\n"
        "1️⃣ Вы отправляете снимок в систему.\n"
        "2️⃣ Алгоритм **MediaPipe Face Landmarker** разворачивает сетку из сотен ключевых точек.\n"
        "3️⃣ Программа вычисляет индекс симметрии и соотношение сторон (вертикаль/горизонталь).\n"
        "4️⃣ Нейросеть **Gemini 2.5 Flash** агрегирует данные и пишет развернутые рекомендации."
    )
    await callback.message.edit_text(info_text, reply_markup=get_back_btn(), parse_mode="Markdown")

@dp.callback_query(F.data == "menu_premium")
async def process_premium(callback: CallbackQuery):
    await callback.answer()
    premium_text = (
        "⭐ **Face Analyzer Premium**\n\n"
        "Откройте расширенные возможности ИИ аналитики:\n"
        "• Отсутствие лимитов на загрузку фото.\n"
        "• Определение фенотипа и этнических черт лица.\n"
        "• Персональный ИИ-стилист по подбору причесок и очков.\n\n"
        "*Функция Premium интеграции находится в разработке.*"
    )
    await callback.message.edit_text(premium_text, reply_markup=get_back_btn(), parse_mode="Markdown")

@dp.callback_query(F.data == "menu_rate")
async def process_rate_photo(callback: CallbackQuery, state: FSMContext):
    await callback.answer()
    await state.set_state(AnalyzerStates.waiting_for_photo)
    
    instruction_text = (
        "📸 **Отправьте фотографию лица**\n\n"
        "⚠️ **Требования к снимку для точной работы ИИ:**\n"
        "• Лицо смотрит строго прямо (анфас)\n"
        "• Хорошее, равномерное освещение\n"
        "• Без медицинских масок и солнцезащитных очков\n"
        "• На фотографии должен быть строго **один человек**"
    )
    await callback.message.edit_text(instruction_text, reply_markup=get_back_btn(), parse_mode="Markdown")

@dp.callback_query(F.data == "menu_history")
async def process_history(callback: CallbackQuery):
    await callback.answer()
    user_id = callback.from_user.id
    
    if not supabase:
        await callback.message.edit_text("⚠️ Система хранения истории отключена или недоступна.", reply_markup=get_back_btn())
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
            await callback.message.edit_text(
                "📜 **Ваша история пуста**\n\nВы еще не сканировали фотографии в нашем боте.",
                reply_markup=get_back_btn(),
                parse_mode="Markdown"
            )
            return
            
        history_text = "📜 **Ваши последние 5 сканирований:**\n\n"
        for idx, row in enumerate(response.data, 1):
            raw_date = row.get('created_at', '-----')
            formatted_date = raw_date[:10] if len(raw_date) >= 10 else raw_date
            score = row.get('score', 0.0)
            history_text += f"{idx}. **Оценка: ⭐ {score}/10** | Дата: `{formatted_date}`\n"
            
        await callback.message.edit_text(history_text, reply_markup=get_back_btn(), parse_mode="Markdown")
        
    except Exception as e:
        logger.error(f"Error querying history from Supabase for user {user_id}: {e}")
        await callback.message.edit_text("❌ Произошла техническая ошибка при чтении истории.", reply_markup=get_back_btn())

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
        
        await message.reply(report, reply_markup=get_back_btn())
        await state.clear()

    except Exception as e:
        logger.error(f"Critical error during photo processing chain: {e}", exc_info=True)
        await bot.delete_message(chat_id=message.chat.id, message_id=waiting_msg.message_id)
        await message.answer(
            f"❌ **Сбой внутренней обработки данных**\n\nПроизошла непредвиденная ошибка на сервере.\nДетали: `{str(e)}`",
            reply_markup=get_back_btn(),
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
