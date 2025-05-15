import os
import json
import requests
import numpy as np
import concurrent.futures
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
from sklearn.metrics.pairwise import cosine_distances
from langchain.schema import Document

# LangChain
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
from langchain_gigachat import GigaChat
from langchain.chains import RetrievalQA
from langchain.schema import AIMessage
from langchain.prompts import PromptTemplate

import os
import fitz
from transformers import AutoTokenizer
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_gigachat import GigaChat

import tempfile
import shutil
from langchain.schema import Document

from config import BOT_TOKEN, AUTH

from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor
import phoenix as px

tracer_provider = register()
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
px.launch_app()

def get_token():
    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json',
        'RqUID': '450ce936-9e4b-4e1c-aa70-a197e3890e8f',
        'Authorization': f'Basic {AUTH}'
    }
    payload = "scope=GIGACHAT_API_CORP"
    response = requests.post(url, headers=headers, data=payload, verify=False)
    return response.json()['access_token']

session = requests.Session()
global_token = get_token()

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("ai-forever/sbert_large_nlu_ru")

def truncate_to_token_limit(text: str, max_tokens: int = 500) -> str:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)

def embeddings(texts, model="Embeddings"):
    url = "https://gigachat.devices.sberbank.ru/api/v1/embeddings"
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f'Bearer {global_token}'
    }

    results = []
    batch_size = 5

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        safe_batch = []

        for t in batch:
            if isinstance(t, str) and t.strip():
                trimmed = truncate_to_token_limit(t)
                safe_batch.append(trimmed)

        if not safe_batch:
            continue

        payload = json.dumps({"model": model, "input": safe_batch})
        response = session.post(url, headers=headers, data=payload, verify=False)

        if response.status_code != 200:
            raise Exception(f"Ошибка в embeddings: {response.status_code}: {response.text}")

        results.extend(response.json()["data"])

    return results



def compare_news(s1, s2):
    embs = embeddings([s1, s2])
    vect1 = np.array(embs[0]["embedding"]).reshape(1, -1)
    vect2 = np.array(embs[1]["embedding"]).reshape(1, -1)
    return cosine_distances(vect1, vect2)[0][0]


def process_candidate(candidate, query):
    try:
        dist = compare_news(query, candidate)
        return candidate, dist
    except Exception as e:
        print(f"Ошибка с '{candidate}': {e}")
        return candidate, float("inf")


def get_best_match_parallel(query, candidates, max_workers=30):
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_candidate, c, query): c for c in candidates}
        for future in concurrent.futures.as_completed(futures):
            candidate, dist = future.result()
            results.append((candidate, dist))
    results.sort(key=lambda x: x[1])
    return results[0][0]


def extract_title_from_pdf(path):
    doc = fitz.open(path)
    lines = doc[0].get_text().split('\n')
    return lines[0].strip() if lines else "Без названия"

def find_best_contract_template(user_query: str, data_root="data") -> str:

    import difflib

    print(f"📁 Ищу лучшую папку среди: {os.listdir(data_root)}")
    folders = [f for f in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, f))]
    if not folders:
        raise ValueError("❗ Нет папок с шаблонами в указанной директории.")

    # 1. Поиск подходящей папки
    folder_scores = []
    for folder in folders:
        try:
            dist = compare_news(user_query, folder)
            folder_scores.append((folder, dist))
        except Exception as e:
            print(f"⚠️ Ошибка при сравнении с папкой '{folder}': {e}")
            continue

    folder_scores.sort(key=lambda x: x[1])
    best_folder = folder_scores[0][0]
    folder_path = os.path.join(data_root, best_folder)
    print(f"📁 Лучшая папка: {folder_path}")

    # 2. Поиск топ-5 файлов в папке
    files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    if not files:
        raise ValueError("❗ В выбранной папке нет PDF-файлов.")

    file_scores = []
    for f in files:
        try:
            dist = compare_news(user_query, f)
            file_scores.append((f, dist))
        except Exception as e:
            print(f"⚠️ Ошибка при сравнении с файлом '{f}': {e}")
            continue

    file_scores.sort(key=lambda x: x[1])
    top_files = file_scores[:5]
    options_block = "\n".join([f"{i+1}. {f[0]}" for i, f in enumerate(top_files)])

    print("📄 Топ-5 шаблонов:")
    print(options_block)

    # 3. Вопрос в GigaChat
    prompt = PromptTemplate(
    input_variables=["contract_text", "options"],
    template="""
Ты помощник, который помогает выбрать лучший шаблон договора. Вот запрос пользователя:

"{contract_text}"

Вот список 5 наиболее близких шаблонов:

{options}

Если в списке нет подходящего шаблона, просто ответь:
"Пока у меня нет такого шаблона. Попробуй позже."

В любом случае, ответь точным названием файла из списка или этой фразой.
"""
)


    llm = GigaChat(
        credentials=AUTH,
        verify_ssl_certs=False,
        scope='GIGACHAT_API_CORP',
        model="GigaChat-2-Max",
        profanity_check=False
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.invoke({
        "contract_text": user_query,
        "options": options_block
    })

    result_text = result["text"].strip()

    # Если GigaChat честно ответил, что ничего не подходит — возвращаем флаг и сообщение
    if result_text.startswith("Пока у меня нет такого шаблона"):
        print("🤖 Ответ от GigaChat: шаблона нет.")
        return {
            "status": "not_found",
            "message": "Пока у меня нет такого шаблона. Попробуй позже."
        }

    # Иначе — ищем ближайшее совпадение
    all_filenames = [f[0] for f in top_files]
    closest_match = difflib.get_close_matches(result_text, all_filenames, n=1, cutoff=0.5)

    if closest_match:
        selected_file = closest_match[0]
        final_path = os.path.join(folder_path, selected_file)
        print(f"📌 GigaChat выбрал: {final_path}")
        return {
            "status": "ok",
            "path": final_path
        }
    else:
        raise FileNotFoundError(f"❗ GigaChat выбрал файл '{result_text}', но он не найден среди топ-5.")

class GigaChatEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        safe_texts = []
        for t in texts:
            if isinstance(t, Document):
                t = t.page_content
            if not isinstance(t, str):
                continue
            t = t.strip()
            if not t:
                continue
            safe_texts.append(t[:1500])  # усечённый текст
        return [item["embedding"] for item in embeddings(safe_texts)]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

def answer_question_about_contract(pdf_path: str, question: str) -> str:
    from langchain.prompts import PromptTemplate

    # Загрузка и очистка PDF
    loader = PyMuPDFLoader(pdf_path)
    pages = loader.load()

    for doc in pages:
        doc.page_content = doc.page_content.replace("\n", " ")

    # Делим на чанки
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(pages)

    # Векторизация
    embedding_model = GigaChatEmbeddings()
    with tempfile.TemporaryDirectory() as chroma_dir:
        vector_db = Chroma.from_documents(chunks, embedding_model, persist_directory=chroma_dir)
        retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 10})  # 👈 ищем больше чанков
        relevant_docs = retriever.get_relevant_documents(question)

    # Косинусная фильтрация (ослаблена)
    question_emb = np.array(embedding_model.embed_query(question)).reshape(1, -1)
    accepted_docs = []
    print("\n🔍 Косинусные расстояния для вопроса:", question)
    for i, doc in enumerate(relevant_docs):
        if not isinstance(doc, Document):
            continue
        doc_text = doc.page_content.strip()
        if not doc_text:
            continue
        doc_emb = np.array(embedding_model.embed_query(doc_text)).reshape(1, -1)
        distance = cosine_distances(question_emb, doc_emb)[0][0]
        print(f"[{i}] Расстояние: {distance:.4f} | Текст: {doc_text[:120].replace(chr(10), ' ')}...")
        if distance < 0.35:  # 👈 более мягкий фильтр
            accepted_docs.append(Document(page_content=doc_text, metadata=doc.metadata or {}))

    if not accepted_docs:
        print("⚠️ Не найдено подходящих чанков по косинусной метрике — передаём всё в LLM")
        accepted_docs = relevant_docs  # 👈 используем всё, что вернул retriever

    # Повторная загрузка отфильтрованных документов
    with tempfile.TemporaryDirectory() as filtered_dir:
        filtered_db = Chroma.from_documents(accepted_docs, embedding_model, persist_directory=filtered_dir)
        retriever_filtered = filtered_db.as_retriever(search_type="mmr", search_kwargs={"k": 5})

        llm = GigaChat(
            credentials=AUTH,
            verify_ssl_certs=False,
            scope='GIGACHAT_API_CORP',
            model="GigaChat-2-Max",
            profanity_check=False
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever_filtered,
            chain_type="stuff",
            return_source_documents=True,
            verbose=False,
            chain_type_kwargs={
                "prompt": PromptTemplate(
                    input_variables=["context", "question"],
                    template="""
Ты помощник по договору. Отвечай только на основании контекста.
Если ответа нет — просто скажи: "Информация не найдена в договоре."

Контекст:
{context}

Вопрос: {question}
Ответ:"""
                )
            }
        )

        response = qa_chain.invoke({"query": question})
        return response["result"]


def legal_audit_from_gk(pdf_path: str, kodex_dir: str = "kodex") -> str:
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain.chains import LLMChain
    import tempfile

    # Загрузка и разбиение договора
    loader = PyMuPDFLoader(pdf_path)
    contract_pages = loader.load()
    for doc in contract_pages:
        doc.page_content = doc.page_content.replace("\n", " ")
    contract_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    contract_chunks = contract_splitter.split_documents(contract_pages)

    # Загрузка и разбиение ГК
    all_kodex_docs = []
    kodex_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    for filename in sorted(os.listdir(kodex_dir)):
        if filename.endswith(".pdf"):
            path = os.path.join(kodex_dir, filename)
            kodex_loader = PyMuPDFLoader(path)
            kodex_pages = kodex_loader.load()
            for doc in kodex_pages:
                doc.page_content = doc.page_content.replace("\n", " ")
            all_kodex_docs.extend(kodex_splitter.split_documents(kodex_pages))

    print(f"📑 Договор: {len(contract_chunks)} чанков | ГК: {len(all_kodex_docs)} статей")

    # Векторизация
    embedding_model = GigaChatEmbeddings()

    with tempfile.TemporaryDirectory() as temp_dir:
        kodex_db = Chroma.from_documents(
            all_kodex_docs, embedding_model, persist_directory=os.path.join(temp_dir, "kodex")
        )
        retriever = kodex_db.as_retriever(search_type="mmr", search_kwargs={"k": 8})

        # Модель
        llm = GigaChat(
            credentials=AUTH,
            verify_ssl_certs=False,
            scope='GIGACHAT_API_CORP',
            model="GigaChat-2-Max",
            profanity_check=False
        )

        # Промпт + цепочка
        prompt = PromptTemplate(
            input_variables=["context", "contract"],
            template="""
Ты — опытный юрист. Проанализируй, содержит ли данный договор нарушения положений Гражданского кодекса РФ.

Контекст (статьи ГК РФ):
{context}

Фрагменты договора:
{contract}

Выведи возможные нарушения и ссылки на статьи. Если всё в порядке, напиши: "Нарушений не выявлено."
"""
        )
        qa_chain = LLMChain(llm=llm, prompt=prompt)

        # Анализ по чанкам
        violations = []
        for i in range(0, len(contract_chunks), 3):
            partial_contract_text = truncate_to_token_limit(
                " ".join(c.page_content for c in contract_chunks[i:i + 3]), 450
            )
            relevant_articles = retriever.get_relevant_documents(partial_contract_text)
            context_text = truncate_to_token_limit(
                "\n\n".join(d.page_content for d in relevant_articles), 800
            )

            result = qa_chain.invoke({
                "context": context_text,
                "contract": partial_contract_text
            })

            violations.append(f"🔹 Фрагмент {i + 1}–{i + 3}:\n{result}\n")

        # Финальный отчёт
        final_report = "\n".join(violations)
        report_path = "gk_audit_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(final_report)

        return report_path  # путь к файлу — для отправки
    
def detect_contract_topic_gigachat(pdf_path: str, topics: List[str]) -> str:
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain

    # 1. Извлекаем текст договора (усекаем по токенам)
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text() for page in doc)
    trimmed_text = truncate_to_token_limit(text, max_tokens=900)

    # 2. Готовим список тем как одну строку
    options_block = "\n".join([f"- {t}" for t in topics])

    # 3. Промпт
    prompt = PromptTemplate(
        input_variables=["contract_text", "options"],
        template="""
Ты — юридический ассистент. Твоя задача — определить, к какому из нижеперечисленных типов договоров относится приведённый ниже текст.

Выбери **только один наиболее подходящий** вариант из списка. Если подходящих нет, напиши: "Не удалось определить".

Типы договоров:
{options}

Текст договора:
{contract_text}

Ответ:
"""
    )

    # 4. Инициализируем GigaChat
    llm = GigaChat(
        credentials=AUTH,
        verify_ssl_certs=False,
        scope='GIGACHAT_API_CORP',
        model="GigaChat-2-Max",
        profanity_check=False
    )

    # 5. Цепочка и запуск
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.invoke({
        "contract_text": trimmed_text,
        "options": options_block
    })
    
    print(result)
    return result["text"].strip() 



# Один токенизатор на все
tokenizer = AutoTokenizer.from_pretrained("ai-forever/sbert_large_nlu_ru")

def truncate(text: str, max_tokens: int = 800) -> str:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)

# Мапа: тема → файл
TOPIC_TO_KODEX_PDF = {
    "Образцы договоров аренды": "Статьи_договора_аренды_квартиры,_нежилого_помещения,_гаража,_земли.pdf",
    "Образцы договоров аренды квартиры": "Статьи_договора_аренды_квартиры,_нежилого_помещения,_гаража,_земли.pdf",
    "Договоры аренды комнаты": "Статьи договора аренды квартиры, нежилого помещения, гаража, земли.pdf",
    "Договоры аренды гаража": "Статьи договора аренды квартиры, нежилого помещения, гаража, земли.pdf",
    "Договоры аренды нежилого помещения": "Статьи договора аренды квартиры, нежилого помещения, гаража, земли.pdf",
    "Договоры аренды торгового помещения": "Статьи договора аренды квартиры, нежилого помещения, гаража, земли.pdf",
    "Договоры аренды зданий": "Статьи договора аренды квартиры, нежилого помещения, гаража, земли.pdf",
    "Договоры аренды земельного участка": "Статьи договора аренды квартиры, нежилого помещения, гаража, земли.pdf",
    "Договоры аренды машиноместа": "Статьи договора аренды оборудования, спецтехники, машиноместа.pdf",
    "Договоры аренды недвижимости": "Статьи договора аренды квартиры, нежилого помещения, гаража, земли.pdf",
    "Договоры аренды спецтехники": "Статьи договора аренды оборудования, спецтехники, машиноместа.pdf",
    "Договоры аренды оборудования": "Статьи договора аренды оборудования, спецтехники, машиноместа.pdf",
    "Договоры аренды автомобиля": "Статьи договора аренды оборудования, спецтехники, машиноместа.pdf",
    "Договоры аренды с правом выкупа": "Статьи договора аренды с правом выкупа.pdf",
    "Образцы договоров субаренды": "Статьи_договора_аренды_квартиры,_нежилого_помещения,_гаража,_земли.pdf",
    "Образцы договоров купли-продажи": "Статьи договора купли-продажи.pdf",
    "Договоры купли-продажи автомобиля": "Статьи договора купли-продажи.pdf",
    "Образцы договоров подряда": "Статьи договора Гпх.pdf",
    "Образцы договоров оказания услуг": "Статьи договора Гпх.pdf",
    "Образцы договоров ГПХ": "Статьи договора Гпх.pdf",
    "Агентские договоры: образцы": "Статьи договора Гпх.pdf",
    "Шаблоны договоров поставки": "Статьи договора Гпх.pdf",
    "Шаблоны договоров цессии": "Статьи договора Гпх.pdf",
    "Договоры сотрудничества": "Статьи договора Гпх.pdf",
    "Шаблоны договоров авторского заказа": "Статьи договора Гпх.pdf",
    "Образцы договоров займа": "Статьи займ-расписка.pdf",
    "Расписки 2025": "Статьи займ-расписка.pdf",
    "Образцы договоров поручительства": "Статьи договора поручительства.pdf",
    "Образцы договоров страхования": "Статьи договора Гпх.pdf",
    "Образцы договоров дарения": "Статьи дарения.pdf",
    "Образцы договоров ссуды": "Статьи договора ссуда.pdf",
    "Образцы договоров хранения": "Статьи договора хранения.pdf",
    "Образцы договоров управления имуществом": "Статьи договора доверительное управление.pdf",
    "Комиссионные договоры: образцы и шаблоны": "Статьи договора Гпх.pdf",
    "Договоры коммерческой концессии": "Статьи договора Гпх.pdf",
    "Образцы договоров лизинга": "Статьи договора Гпх.pdf",
    "Образцы лицензионных договоров": "Статьи договора лицензии.pdf",
    "Брачные договоры: образцы и примеры": "Статьи брачного договора.pdf",
    "Образцы трудовых договоров": "Статьи_договора_найма_работника.pdf",
    "Договоры найма: образцы и шаблоны": "Статьи договора найма работника.pdf",
    "Образцы договоров перевозки": "Статьи договора Гпх.pdf",
    "Документы для банкротства физического лица": "Статьи займ-расписка.pdf"
}

def check_contract_by_detected_topic(contract_pdf_path: str, topic: str) -> str:
    kodex_filename = TOPIC_TO_KODEX_PDF.get(topic)
    if not kodex_filename:
        raise ValueError(f"❗ Нет PDF статей для темы: {topic}")
    
    kodex_path = os.path.join("kodex", kodex_filename)
    if not os.path.exists(kodex_path):
        raise FileNotFoundError(f"❗ Не найден файл: {kodex_path}")

    contract_text = "\n".join(page.get_text() for page in fitz.open(contract_pdf_path))
    gk_text = "\n".join(page.get_text() for page in fitz.open(kodex_path))

    # Делим на статьи
    articles = []
    current = []
    title = None
    for line in gk_text.splitlines():
        if line.strip().startswith("Статья"):
            if title and current:
                articles.append((title, "\n".join(current)))
            title = line.strip()
            current = []
        else:
            current.append(line)
    if title and current:
        articles.append((title, "\n".join(current)))

    llm = GigaChat(
        credentials=AUTH,
        verify_ssl_certs=False,
        scope="GIGACHAT_API_CORP",
        model="GigaChat-2-Max",
        profanity_check=False
    )

    prompt = PromptTemplate(
        input_variables=["article", "contract"],
        
        template = '''
    Ты — квалифицированный юрист с опытом договорной и корпоративной практики. 
Проанализируй, соответствует ли **фрагмент договора** требованиям указанной статьи законодательства.

1. Если статья относится к **Гражданскому кодексу РФ**, но текст — например, из **трудового договора**, укажи, что следует руководствоваться Трудовым кодексом РФ, а не ГК РФ.
2. Если статья корректна по предмету, проведи **юридическое сравнение**: соответствие, противоречия, пробелы.
3. Нарушения формулируй **чётко**, с пояснением, почему они нарушают норму (например: "Статья 689 требует письменной формы — в тексте её нет").
4. Структурируй ответ: **"Соответствие / Нарушения / Рекомендации"**.
5. В конце дай краткий **юридический вывод**: "Нарушений не выявлено" или "Обнаружены существенные нарушения".

Исходные данные:

📘 Статья (ссылка на норму):
{article}

📄 Фрагмент договора:
{contract}

⚖️ Ответ:
'''
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    trimmed_contract = truncate(contract_text)

    results = []
    for title, content in articles:
        print(f"\n📘 {title}:\n{content[:300]}...\n{'-'*40}")
        result = chain.invoke({
            "article": truncate(content),
            "contract": trimmed_contract
        })
        results.append(f"🔹 {title}:\n{result['text'].strip()}\n")

    '''out_path = "gk_topic_audit.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(results))'''
    
    full_audit_path = "gk_topic_audit.txt"
    with open(full_audit_path, "w", encoding="utf-8") as f:
        f.write("\n".join(results))

    # Сохраняем краткое саммари
    summary_prompt = PromptTemplate(
        input_variables=["full_audit"],
        template="""
Ты — опытный юрист. У тебя на руках большой анализ соответствия договора нормам Гражданского кодекса РФ. На основе него:

1. Сформулируй краткое юридическое **саммари на 1–2 страницы** (1500–2000 символов).
2. Структура:
- **Основные выявленные нарушения**
- **Риски (если оставить как есть)**
- **Краткие рекомендации**
- **Юридический вывод**
3. Пиши юридически чётко, без воды, без повторов.

📄 Анализ:
{full_audit}

✂️ Краткое юридическое заключение:
"""
    )

    summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
    summary_result = summary_chain.invoke({"full_audit": "\n".join(results)})

    summary_path = "gk_topic_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_result["text"].strip())

    # Вернуть оба пути
    return full_audit_path, summary_path