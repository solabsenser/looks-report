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

from analyzer import ANALYZER_BACKEND, analyze_face
from premium import (
    generate_mesh_overlay,
    generate_heatmap,
    generate_debug_overlay,
    generate_premium_report
)
from aiogram.types import FSInputFile
from aiogram.types import LabeledPrice
from aiogram.types import PreCheckoutQuery

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

# ===== ADMIN ========
ADMIN_IDS = [
    int(admin_id)
    for admin_id in os.getenv(
        "ADMIN_IDS",
        ""
    ).split(",")
    if admin_id.strip()
]

# ===== PREMIUM =====
async def is_premium_user(user_id: int):
    try:
        response = (
            supabase
            .table("leaderboard")
            .select("is_premium")
            .eq("user_id", user_id)
            .single()
            .execute()
        )

        return response.data.get("is_premium", False)

    except:
        return False


async def activate_premium(user_id: int):
    (
        supabase
        .table("leaderboard")
        .update({
            "is_premium": True
        })
        .eq("user_id", user_id)
        .execute()
    )
    
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

# ======= PREMIUM BUY =========
@dp.message(F.text == "💎 Премиум")
async def process_premium(message: Message, state: FSMContext):
    await state.clear()

    premium = await is_premium_user(
        message.from_user.id
    )

    if premium:
        await message.answer(
            "⭐ Premium уже активирован!\n\n"
            "Ваши возможности:\n\n"
            "✅ Face Heatmap\n"
            "✅ FaceMesh Visualization\n"
            "✅ Debug Analysis Mode\n"
            "✅ Все будущие Premium обновления\n\n"
            "Статус: Активен 🟢"
        )
        return

    buy_keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="💳 Купить Premium",
                    callback_data="buy_premium"
                )
            ]
        ]
    )

    await message.answer(
        "💎 FACE ANALYZER PREMIUM\n\n"

        "Что открывает Premium:\n\n"

        "🗺 Face Heatmap\n"
        "Визуализация сильных и слабых зон лица.\n\n"

        "🧠 FaceMesh Analysis\n"
        "Все точки анализа MediaPipe поверх фотографии.\n\n"

        "📐 Debug Visualization\n"
        "Показ измерений симметрии, пропорций и структуры лица.\n\n"

        "📊 Extended Report\n"
        "Расширенный отчет со всеми вычисленными метриками.\n\n"

        "🚀 Early Access\n"
        "Доступ к новым функциям раньше остальных пользователей.\n\n"

        "⭐ Стоимость: 10 Stars",
        reply_markup=buy_keyboard
    )

@dp.callback_query(F.data == "buy_premium")
async def buy_premium(callback: CallbackQuery):

    premium = await is_premium_user(
        callback.from_user.id
    )

    if premium:

        await callback.answer(
            "Premium уже активирован",
            show_alert=True
        )
        return

    try:
        await callback.message.delete()
    except:
        pass

    await callback.message.answer_invoice(
        title="Face Analyzer Premium",
        description=(
            "Premium доступ:\n"
            "• Heatmap\n"
            "• FaceMesh\n"
            "• Debug Visualization\n"
            "• Extended Report"
        ),
        payload="premium_forever",
        currency="XTR",
        prices=[
            LabeledPrice(
                label="Premium Forever",
                amount=10
            )
        ]
    )

    await callback.answer()

@dp.pre_checkout_query()
async def process_pre_checkout(
    pre_checkout_query: PreCheckoutQuery
):
    await bot.answer_pre_checkout_query(
        pre_checkout_query.id,
        ok=True
    )

@dp.message(F.successful_payment)
async def successful_payment(
    message: Message
):

    if (
        message.successful_payment.invoice_payload
        != "premium_forever"
    ):
        return

    # Защита от повторной покупки
    already_premium = await is_premium_user(
        message.from_user.id
    )

    if already_premium:

        try:
            await bot.delete_message(
                chat_id=message.chat.id,
                message_id=message.message_id
            )
        except:
            pass

        await bot.send_message(
            message.chat.id,
            "⚠️ Premium уже активирован на вашем аккаунте."
        )
        return

    await activate_premium(
        message.from_user.id
    )

    try:
        await bot.delete_message(
            chat_id=message.chat.id,
            message_id=message.message_id
        )
    except Exception as e:
        logger.error(
            f"Failed to delete invoice: {e}"
        )

    await bot.send_message(
        message.chat.id,
        "🎉 Premium успешно приобретён!\n\n"
        "Спасибо за поддержку Face Analyzer ❤️\n\n"
        "💎 Ваш Premium активирован навсегда.\n\n"
        "Теперь вам доступны:\n\n"
        "✅ Face Heatmap\n"
        "✅ FaceMesh Visualization\n"
        "✅ Debug Analysis Mode\n"
        "✅ Extended Report\n"
        "✅ Все будущие Premium обновления\n\n"
        "⚠️ Premium уже привязан к вашему аккаунту.\n"
        "Повторная покупка не требуется.\n\n"
        "Приятного использования и удачного анализа! 🚀"
    )

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
        if supabase:
            try:
                current_photo_id = photo.file_id
                # Ищем запись по ID пользователя
                user_check = supabase.table("leaderboard").select("*").eq("user_id", user_id).execute()
                
                if user_check.data:
                    record = user_check.data[0]
                    old_max = float(record.get("max_score", 0.0))
                    old_streak = int(record.get("streak", 1))
                    last_photo_id = record.get("last_photo_id", "")
                    
                    if last_photo_id == current_photo_id:
                        milestone_text = "⚠️ <b>Стрик не засчитан!</b> Вы отправили то же самое фото."
                    
                    elif score > old_max:
                        # Новый рекорд — обновляем
                        supabase.table("leaderboard").update({
                            "max_score": score,
                            "last_photo_id": current_photo_id,
                            "updated_at": datetime.now().isoformat()
                        }).eq("user_id", user_id).execute()
                        milestone_text = f"👑 <b>Новый рекорд!</b> Ваша оценка: <b>{score}/10</b>."
                    
                    elif abs(score - old_max) <= 0.5:
                        # Подтверждение уровня — обновляем стрик
                        new_streak = old_streak + 1
                        supabase.table("leaderboard").update({
                            "streak": new_streak,
                            "last_photo_id": current_photo_id,
                            "updated_at": datetime.now().isoformat()
                        }).eq("user_id", user_id).execute()
                        milestone_text = f"🔥 <b>Стрик х{new_streak}!</b> Ваш уровень стабилен ({score}/10)."
                    
                    else:
                        milestone_text = f"✅ Анализ: <b>{score}/10</b>. (Ваш рекорд: {old_max}/10)."
                
                else:
                    # Новая запись — создаем
                    supabase.table("leaderboard").insert({
                        "user_id": user_id,
                        "username": username,
                        "max_score": score,
                        "streak": 1,
                        "last_photo_id": current_photo_id,
                        "updated_at": datetime.now().isoformat()
                    }).execute()
                    milestone_text = f"🏆 <b>Добро пожаловать в таблицу!</b> Результат <b>{score}/10</b> зафиксирован."
                            
            except Exception as db_err:
                logger.error(f"DB Error: {db_err}")
                milestone_text = "❌ Ошибка записи в базу."
                
        # Удаляем сообщение загрузки
        await bot.delete_message(
            chat_id=message.chat.id,
            message_id=waiting_msg.message_id
        )

        # ===== PREMIUM =====

        premium = await is_premium_user(
            user_id
        )

        if premium:

            try:

                landmarks = result["landmarks"]
                scores = result["scores"]

                mesh_file = generate_mesh_overlay(
                    temp_image_path,
                    landmarks
                )

                heatmap_file = generate_heatmap(
                    temp_image_path,
                    landmarks,
                    scores
                )

                debug_file = generate_debug_overlay(
                    temp_image_path,
                    landmarks
                )

                premium_file = generate_premium_report(
                    temp_image_path,
                    mesh_file,
                    heatmap_file,
                    debug_file
                )

                if premium_file:

                    await message.answer_photo(
                        FSInputFile(premium_file),
                        caption="💎 Premium Face Analysis"
                    )

                for file_path in [
                    mesh_file,
                    heatmap_file,
                    debug_file,
                    premium_file
                ]:

                    if (
                        file_path
                        and os.path.exists(file_path)
                    ):
                        os.remove(file_path)

            except Exception as premium_error:

                logger.error(
                    f"Premium error: {premium_error}"
                )
                
        # Отправляем отчет
        await message.reply(
            report,
            parse_mode="HTML"
        )
        
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

@dp.message(Command("givepremium"))
async def give_premium_admin(
    message: Message
):

    if message.from_user.id not in ADMIN_IDS:
        return

    args = message.text.split()

    if len(args) != 2:

        await message.answer(
            "Использование:\n"
            "/givepremium USER_ID"
        )
        return

    try:

        target_user = int(args[1])

        await activate_premium(
            target_user
        )

        await message.answer(
            f"✅ Premium выдан пользователю {target_user}"
        )

        try:

            await bot.send_message(
                target_user,
                "🎉 Администратор активировал вам Premium!\n\n"
                "💎 Premium активирован навсегда."
            )

        except:
            pass

    except Exception as e:

        await message.answer(
            f"❌ Ошибка: {e}"
        )
    
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
