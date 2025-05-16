import os
import asyncio
from aiogram import Bot, Dispatcher, F, types
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from dotenv import load_dotenv
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from functions_for_agent import (
    answer_question_about_contract,
    find_best_contract_template,
    legal_audit_from_gk  # 👈 добавлено
)
from config import BOT_TOKEN

load_dotenv()
PDF_PATH = "dogovors/latest.pdf"
TEMPLATE_PATH = "dogovors/temp_template.pdf"

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(storage=MemoryStorage())


# Состояния
class Form(StatesGroup):
    waiting_for_question = State()
    waiting_for_template_query = State()
    waiting_for_audit = State()


# Клавиатура
menu_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="Найти инфу в договоре")],
        [KeyboardButton(text="Получить шаблон")],
        [KeyboardButton(text="Проверить на нарушения в ГК РФ")]
    ],
    resize_keyboard=True
)


# Команда /start
@dp.message(F.text == "/start")
async def on_start(message: Message, state: FSMContext):
    await state.clear()

    # Удаляем все старые файлы
    folder = "dogovors"
    for f in os.listdir(folder):
        try:
            os.remove(os.path.join(folder, f))
        except Exception as e:
            print(f"Ошибка при удалении {f}: {e}")

    await message.answer("Привет! Пришлите PDF-документ договора или выберите действие ниже:", reply_markup=menu_keyboard)


# Обработка PDF
@dp.message(F.document)
async def on_pdf(message: Message, state: FSMContext):
    if not message.document.file_name.endswith(".pdf"):
        await message.answer("Пожалуйста, отправьте PDF-файл.")
        return
    await bot.download(message.document, destination=PDF_PATH)
    await message.answer("📄 Файл получен. Выберите, что вы хотите сделать:", reply_markup=menu_keyboard)


# Найти инфу
@dp.message(F.text == "Найти инфу в договоре")
async def on_search_info(message: Message, state: FSMContext):
    if not os.path.exists(PDF_PATH):
        await message.answer("Сначала отправьте PDF-документ.")
        return
    await message.answer("✏️ Напишите вопрос, на который хотите найти ответ в договоре:")
    await state.set_state(Form.waiting_for_question)


@dp.message(Form.waiting_for_question)
async def on_question_input(message: Message, state: FSMContext):
    try:
        await message.answer("Ищу ответ в договоре...")
        await message.answer("⏳")
        answer = answer_question_about_contract(PDF_PATH, message.text)
        await message.answer(f"Ответ:\n{answer}")
    except Exception as e:
        await message.answer(f"❗ Ошибка: {str(e)}")
    await state.clear()


# Получить шаблон
@dp.message(F.text == "Получить шаблон")
async def on_get_template(message: Message, state: FSMContext):
    await message.answer("Напишите, какой шаблон договора вы ищете:")
    await state.set_state(Form.waiting_for_template_query)


@dp.message(Form.waiting_for_template_query)
async def on_template_request(message: Message, state: FSMContext):
    try:
        await message.answer("Ищу подходящий шаблон...")
        await message.answer("⏳")

        result = find_best_contract_template(user_query=message.text)

        if result["status"] == "ok":
            import shutil
            shutil.copyfile(result["path"], TEMPLATE_PATH)

            await message.answer_document(
                types.FSInputFile(TEMPLATE_PATH),
                caption="Вот лучший найденный шаблон.\n\nТеперь вы можете загрузить ваш договор, чтобы задать к нему вопросы."
            )
        else:
            # шаблона нет — просто отправляем сообщение от GigaChat
            await message.answer(result["message"])

    except Exception as e:
        await message.answer(f"❗ Ошибка при поиске шаблона: {str(e)}")

    await state.clear()

TOPICS = [
    "Образцы договоров аренды",
    "Образцы договоров аренды квартиры",
    "Договоры аренды комнаты",
    "Договоры аренды гаража",
    "Договоры аренды нежилого помещения",
    "Договоры аренды торгового помещения",
    "Договоры аренды зданий",
    "Договоры аренды земельного участка",
    "Договоры аренды машиноместа",
    "Договоры аренды недвижимости",
    "Договоры аренды спецтехники",
    "Договоры аренды оборудования",
    "Договоры аренды автомобиля",
    "Договоры аренды с правом выкупа",
    "Образцы договоров субаренды",
    
    "Образцы договоров купли-продажи",
    "Договоры купли-продажи автомобиля",
    
    "Образцы договоров подряда",
    "Образцы договоров оказания услуг",
    "Образцы договоров ГПХ",
    "Агентские договоры: образцы",
    "Шаблоны договоров поставки",
    "Шаблоны договоров цессии",
    "Договоры сотрудничества",
    "Шаблоны договоров авторского заказа",
    
    "Образцы договоров займа",
    "Расписки 2025",
    
    "Образцы договоров поручительства",
    "Образцы договоров страхования",
    "Образцы договоров дарения",
    "Образцы договоров ссуды",
    "Образцы договоров хранения",
    "Образцы договоров управления имуществом",
    "Комиссионные договоры: образцы и шаблоны",
    "Договоры коммерческой концессии",
    "Образцы договоров лизинга",
    "Образцы лицензионных договоров",
    "Брачные договоры: образцы и примеры",
    
    "Образцы трудовых договоров",
    "Договоры найма: образцы и шаблоны",
    
    "Образцы договоров перевозки",
    
    "Документы для банкротства физического лица"
]

@dp.message(F.text == "Проверить на нарушения в ГК РФ")
async def on_check_by_detected_topic(message: Message, state: FSMContext):
    if not os.path.exists(PDF_PATH):
        await message.answer("Сначала отправьте PDF-договор.")
        return

    await message.answer("Определяю тему договора и запускаю проверку по ГК...")
    await message.answer("⏳")

    try:
        from functions_for_agent import detect_contract_topic_gigachat, check_contract_by_detected_topic

        topic = detect_contract_topic_gigachat(PDF_PATH, TOPICS)
        await message.answer(f"Определено: *{topic}*", parse_mode="Markdown")

        audit_path, summary_path = check_contract_by_detected_topic(PDF_PATH, topic)

        if os.path.exists(audit_path):
            await message.answer_document(types.FSInputFile(audit_path), caption="📎 Подробный отчёт по договору")

        if os.path.exists(summary_path):
            await message.answer_document(types.FSInputFile(summary_path), caption="📄 Краткое юридическое заключение")

    except Exception as e:
        await message.answer(f"❗ Ошибка при проверке: {str(e)}")





# Запуск
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())