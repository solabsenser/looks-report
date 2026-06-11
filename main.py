import os
import asyncio
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton
from aiogram.filters import CommandStart
from dotenv import load_dotenv
from supabase import create_client, Client

# Импортируем функцию анализа из ТВОЕГО файла analyzer.py
from analyzer import analyze_face

# Загружаем переменные окружения
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN not found in .env")

# Инициализация бота и диспетчера
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# Инициализация Supabase (если ключи добавлены в .env)
supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Создаем главное меню с кнопками
main_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="📊 Оценить фото"), KeyboardButton(text="📜 Моя история")]
    ],
    resize_keyboard=True
)

# Хэндлер на команду /start
@dp.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(
        f"Привет, {message.from_user.full_name}! 👋\n"
        "Я бот для анализа внешности. Нажми на кнопку ниже или просто отправь мне свое фото (анфас, хорошее освещение).",
        reply_markup=main_keyboard
    )

# Обработка текстовых кнопок
@dp.message(F.text == "📊 Оценить фото")
async def press_rate_photo(message: Message):
    await message.answer("Отправь мне свою фотографию, и я проведу полный анализ лица!")

@dp.message(F.text == "📜 Моя история")
async def press_history(message: Message):
    if not supabase:
        await message.answer("⚠️ База данных временно недоступна.")
        return

    try:
        # Запрос последних 5 оценок пользователя из Supabase
        response = supabase.table("photo_ratings").select("*").eq("user_id", message.from_user.id).order("created_at", desc=True).limit(5).execute()
        
        if not response.data:
            await message.answer("Вы еще не сканировали фото. Ваша история пуста!")
            return
            
        history_text = "📜 Ваши последние оценки:\n\n"
        for idx, row in enumerate(response.data, 1):
            history_text += f"{idx}. Оценка: ⭐ {row.get('score', 0)}/10 (Дата: {row.get('created_at', '')[:10]})\n"
            
        await message.answer(history_text)
    except Exception as e:
        await message.answer("Не удалось загрузить историю оценок.")
        print(f"Supabase error: {e}")

# Главный хэндлер, который принимает фото
@dp.message(F.photo)
async def handle_photo(message: Message):
    waiting_msg = await message.answer("🔄 Скачиваю и анализирую фото, подожди немного...")
    
    # Берем самое лучшее качество фото
    photo = message.photo[-1]
    
    # Формируем временное имя файла на основе file_id
    temp_image_path = f"temp_{photo.file_id}.jpg"
    
    try:
        # 1. Скачиваем файл из Telegram на диск (как требует твой analyzer.py)
        await bot.download(photo, destination=temp_image_path)
        
        # 2. Вызываем ТВОЮ функцию из analyzer.py
        # Запускаем в executor, чтобы тяжелая синхронная обработка OpenCV не вешала бота
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, analyze_face, temp_image_path)
        
        # 3. Если лицо найдено и оценка успешна, сохраняем в Supabase
        if result["score"] > 0 and supabase:
            try:
                supabase.table("photo_ratings").insert({
                    "user_id": message.from_user.id,
                    "username": message.from_user.username,
                    "score": result["score"],
                    "report": result["report"]
                }).execute()
            except Exception as db_err:
                print(f"Ошибка записи в Supabase: {db_err}")

        # 4. Удаляем "Ожидайте" сообщение и отправляем отчет пользователю
        await bot.delete_message(chat_id=message.chat.id, message_id=waiting_msg.message_id)
        await message.reply(result["report"])
        
    except Exception as e:
        await message.answer(f"❌ Произошла ошибка во время обработки.\nДетали: {e}")
        
    finally:
        # 5. Обязательно удаляем временный файл с диска, чтобы сервер не переполнился
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

# Запуск бота
async def main():
    print("Бот запущен и готов к работе!")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
