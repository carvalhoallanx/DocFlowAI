import os
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

BASE_DIR = Path(__file__).resolve().parents[1]
VECTOR_STORE_DIR = str(BASE_DIR / "vector_store")
DEFAULT_DOCS_DIR = str(BASE_DIR / "doc")


def load_pdf(pdf):
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(pdf)
    docs = list(loader.lazy_load())
    for doc in docs:
        doc.metadata["source"] = pdf
        doc.metadata["page"] = doc.metadata.get("page", 0)
    return docs

def add_documents_to_store(pdf_list):
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
    )

    # 🔁 Verifica se já existe banco
    if os.path.exists(VECTOR_STORE_DIR):
        db = FAISS.load_local(
            VECTOR_STORE_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        db = None

    # 🔁 Verifica se pdf enviados já existem no banco
    existing_sources = set()
    if db:
        existing_sources = set(
            [doc.metadata.get("source") for doc in db.docstore._dict.values()]
        )

    pdf_list = [pdf for pdf in pdf_list if pdf not in existing_sources]

    if not pdf_list:
        print("Nenhum PDF novo para adicionar.")
        return
    
    # 🔁 Carrega os PDFs em paralelo usando ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(load_pdf, pdf_list))

    all_docs = [doc for sublist in results for doc in sublist]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    split_docs = splitter.split_documents(all_docs)

    # 🧠 Criar ou adicionar
    if db:
        db.add_documents(split_docs)
    else:
        db = FAISS.from_documents(split_docs, embeddings)

    db.save_local(VECTOR_STORE_DIR)

    print("✅ Documentos adicionados ao banco!")


def clear_store_and_docs(docs_dir=DEFAULT_DOCS_DIR, vector_store_dir=VECTOR_STORE_DIR):
    if os.path.exists(vector_store_dir):
        shutil.rmtree(vector_store_dir)

    if docs_dir and os.path.exists(docs_dir):
        for file_name in os.listdir(docs_dir):
            file_path = os.path.join(docs_dir, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)