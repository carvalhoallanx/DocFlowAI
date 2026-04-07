import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from concurrent.futures import ThreadPoolExecutor

def load_pdf(pdf):
    loader = PyPDFLoader(pdf)
    docs = list(loader.lazy_load())
    for doc in docs:
        doc.metadata["source"] = pdf
        doc.metadata["page"] = doc.metadata.get("page", 0)
    return docs

def add_documents_to_store(pdf_list):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
    )

    # 🔁 Verifica se já existe banco
    if os.path.exists("vector_store"):
        db = FAISS.load_local(
            "vector_store",
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

    db.save_local("vector_store")

    print("✅ Documentos adicionados ao banco!")