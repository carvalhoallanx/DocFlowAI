import time
import os
from pathlib import Path
import ollama

from rag.ingest import add_documents_to_store
from rag.rag_pipeline import load_db
from rag.insights import save_insight

BASE_DIR = Path(__file__).resolve().parents[1]
WATCH_FOLDER = BASE_DIR / "auto_docs"
PROCESSED_FOLDER = BASE_DIR / "processed_docs"

# Configuração do modelo Ollama - pode ser alterada via variável de ambiente
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")

os.makedirs(WATCH_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

running = False

def start_watcher():
    global running

    if running:
        return

    running = True
    watch_folder()

def generate_insight(file_path):
    db = load_db()

    docs = db.similarity_search("resuma este documento", k=5)

    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
    Analise o documento e gere:

    - resumo claro
    - principais pontos
    - riscos
    - oportunidades

    {context}
    """

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


def process_file(file_path):
    print(f"📄 Processando: {file_path}")

    # adiciona ao banco vetorial
    add_documents_to_store([file_path])

    # gera insight
    insight = generate_insight(file_path)

    # salva insight
    save_insight(file_path, insight)

    # move arquivo para processed
    new_path = PROCESSED_FOLDER / os.path.basename(file_path)
    os.rename(file_path, str(new_path))

    print(f"✅ Finalizado: {file_path}")


def watch_folder():
    print("👀 Monitorando pasta automática...")

    while True:
        files = WATCH_FOLDER.glob("*.pdf")

        for file in files:
            try:
                process_file(str(file))
            except Exception as e:
                print(f"Erro: {e}")

        time.sleep(5)

def generate_insights(db):
    docs = db.similarity_search("resuma os pontos principais", k=10)

    context = "\n\n".join([d.page_content for d in docs])

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{
            "role": "user",
            "content": f"Resuma insights importantes:\n{context}"
        }]
    )

    Path("data/insights.txt").write_text(
        response["message"]["content"],
        encoding="utf-8"
    )