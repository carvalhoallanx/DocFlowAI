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
    from langchain_ollama import OllamaEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # 🔁 Verifica se já existe banco
    if os.path.exists(VECTOR_STORE_DIR):
        db = FAISS.load_local(
            VECTOR_STORE_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        db = None

    # 🔁 Verifica se pdf enviados já existem no banco (nota: verificação simplificada)
    existing_sources = set()
    # Comentado: a verificação de duplicatas será feita no armazenamento
    # pois acessar docstore._dict é instável entre versões do langchain

    # Para esta versão, adicionaremos todos os PDFs fornecidos
    # Se quiser evitar duplicatas, remova PDFs antigos antes de enviar

    if not pdf_list:
        print("Nenhum PDF para adicionar.")
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