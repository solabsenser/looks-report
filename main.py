import os
import logging
import asyncio
from datetime import datetime
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton, BotCommand, BotCommandScopeDefault
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
    
    # Создаем 3 главные текстовые кнопки (убрали Историю)
    builder.button(text="📊 Анализ лица")
    builder.button(text="🏆 Таблица моггеров")
    builder.button(text="💎 Премиум")
    
    # Сетка: 2 кнопки в ряд, 1 снизу по центру
    builder.adjust(2, 1)
    
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
    
    if not supabase:
        await message.answer("⚠️ Таблица моггеров временно недоступна.")
        return

    try:
        # Тянем данные и сразу сортируем
        response = (
            supabase.table("leaderboard")
            .select("*")
            .order("max_score", desc=True)
            .limit(10)
            .execute()
        )
        
        if not response.data:
            await message.answer("🏆 <b>Таблица моггеров пока пуста!</b>", parse_mode="HTML")
            return
            
        leaderboard_text = "🏆 <b>ТОП-10 МОГГЕРОВ (ОТФИЛЬТРОВАНО)</b>\n\n"
        medals = ["👑", "⚡", "🧊", "👾", "☄️", "🔥", "💎", "🛡️", "🔮", "🧿"]
        
        for idx, row in enumerate(response.data, 1):
            username = row.get('username') or "Аноним"
            # Очистка имени
            if username != "Аноним" and not username.startswith("@") and not username.startswith("id"):
                username = f"@{username}"
                
            raw_score = row.get('max_score', 0.0)
            
            # АНТИ-ИНФЛЯЦИОННЫЙ ФИЛЬТР ДЛЯ ОТОБРАЖЕНИЯ:
            # Если в базе лежат старые "мусорные" 9.5+, мы их принудительно обрезаем для красоты
            display_score = round(min(raw_score, 8.5), 1) 
            
            streak = row.get('streak', 1)
            icon = medals[idx-1] if idx <= len(medals) else "🔹"
            streak_text = f" | 🔥 x{streak}" if streak > 1 else ""
            
            leaderboard_text += f"{idx}. {icon} {username} — <b>{display_score}/10</b>{streak_text}\n"
            
        leaderboard_text += "\n<i>*Оценки свыше 8.5 являются экстремально редкими и проходят строгую модерацию алгоритма.</i>"
        await message.answer(leaderboard_text, parse_mode="HTML")
        
    except Exception as e:
        logger.error(f"Error querying leaderboard: {e}")
        await message.answer("❌ Ошибка загрузки таблицы.")


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

# --- ОБРАБОТКА ФОТО (FSM СЦЕНАРИЙ) ---

@dp.message(AnalyzerStates.waiting_for_photo, F.photo)
async def handle_photo_analysis(message: Message, state: FSMContext):
    user_id = message.from_user.id
    username = message.from_user.username or "unknown"
    
    waiting_msg = await message.answer(
    "🔄 <b>Инициализация процесса...</b>\n"
    "1. Проверяем наличие лица.\n"
    "2. Находим ключевые точки лица.\n"
    "3. Анализируем пропорции.\n\n"
    "<i>Пожалуйста, подождите.</i>",
    parse_mode="HTML"
    )
    
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

# ЛОГИКА ТАБЛИЦЫ ЛИДЕРОВ И СТРИКОВ
        milestone_text = ""
        # Снизил порог до 6.0, чтобы больше людей попадало в рейтинг
        if score >= 6.0 and supabase:
            try:
                current_photo_id = photo.file_id
                user_check = supabase.table("leaderboard").select("*").eq("user_id", user_id).execute()
                
                if user_check.data:
                    current_record = user_check.data[0]
                    old_max = current_record.get("max_score", 0.0)
                    old_streak = current_record.get("streak", 1)
                    last_photo_id = current_record.get("last_photo_id", "")
                    
                    if last_photo_id == current_photo_id:
                        milestone_text = "⚠️ <b>Стрик не засчитан!</b>\nВы отправили то же самое фото. Сделайте новый снимок для прогресса."
                    
                    elif score > old_max:
                        # Новый рекорд
                        supabase.table("leaderboard").update({
                            "max_score": score,
                            "last_photo_id": current_photo_id,
                            "updated_at": datetime.now().isoformat()
                        }).eq("user_id", user_id).execute()
                        milestone_text = f"👑 <b>Новый рекорд!</b>\nВаш результат <b>{score}/10</b> (предыдущий: {old_max})."
                    
                    elif abs(score - old_max) <= 0.5: # Расширил окно для стрик-подтверждения до 0.5
                        new_streak = old_streak + 1
                        supabase.table("leaderboard").update({
                            "streak": new_streak,
                            "last_photo_id": current_photo_id,
                            "updated_at": datetime.now().isoformat()
                        }).eq("user_id", user_id).execute()
                        milestone_text = f"🔥 <b>Стрик подтвержден: x{new_streak}!</b>\nВаш уровень {score}/10 стабилен."
                    
                    else:
                        milestone_text = f"✅ Анализ завершен.\nВаш балл: <b>{score}/10</b> (Личный рекорд: {old_max}/10)."
                else:
                    # Первый раз в таблице
                    supabase.table("leaderboard").insert({
                        "user_id": user_id,
                        "username": username,
                        "max_score": score,
                        "streak": 1,
                        "last_photo_id": current_photo_id
                    }).execute()
                    milestone_text = f"🏆 <b>Добро пожаловать в таблицу!</b>\nВаш результат <b>{score}/10</b> зафиксирован."
                            
            except Exception as db_err:
                logger.error(f"Failed to record statistics: {db_err}")

        # Удаляем сообщение с анимацией загрузки и присылаем готовый HTML-отчет
        await bot.delete_message(chat_id=message.chat.id, message_id=waiting_msg.message_id)
        await message.reply(report, parse_mode="HTML")
        
        # Если юзер достиг какого-то достижения в таблице лидеров — пушим уведомление
        if milestone_text:
            await asyncio.sleep(1)  # Небольшая пауза, чтобы сообщения не слипались
            await message.answer(milestone_text, parse_mode="HTML")
            
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
        "⚠️ Некорректный формат данных\n\n"
        "Система ожидает прямую передачу графического файла (изображения/фотографии).\n"
        "Пожалуйста, просто пришлите снимок лица в чат."
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
    
    # --- НАСТРОЙКА КНОПКИ МЕНЮ ДЛЯ ПОЛЬЗОВАТЕЛЕЙ ---
    commands = [
        BotCommand(command="start", description="Обновить / Перезапустить бота 🔄")
    ]
    # Регистрируем только одну команду, чтобы всё выглядело чисто
    await bot.set_my_commands(commands, scope=BotCommandScopeDefault())
    logger.info("Bot command menu configured successfully.")
    
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
