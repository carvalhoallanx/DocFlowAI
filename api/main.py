import os
from pathlib import Path
from typing import Literal
import threading
from rag.automation import watch_folder
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field
from rag.rag_pipeline import generate_study_material, load_db, ask_llm
from rag.memory import save_message
from rag.ingest import add_documents_to_store, clear_store_and_docs

BASE_DIR = Path(__file__).resolve().parents[1]

app = FastAPI(
    title="DocFlow AI",
    description="API para ingestao de PDFs e geracao de materiais de estudo.",
    version="1.0.0",
)

DOCS_DIR = BASE_DIR / "doc"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"

db = load_db()

@app.on_event("startup")
def start_automation():
    thread = threading.Thread(target=watch_folder)
    thread.daemon = True
    thread.start()

class GenerateRequest(BaseModel):
    request_text: str = Field(..., min_length=2, description="Pedido do usuario")
    output_type: Literal[
        "resposta",
        "anotacoes",
        "resumo",
        "mapa_mental",
        "cronograma",
        "questionario",
    ] = "resposta"
    k: int = Field(default=3, ge=1, le=10)


@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(data: dict):
    question = data["question"]
    chat_id = data["chat_id"]

    response, docs = ask_llm(question, db, chat_id)
    
    save_message(chat_id, "user", question)

    return {"status": "streaming"}  # frontend consome stream

@app.post("/documents/upload")
async def upload_documents(files: list[UploadFile] = File(...)):
    DOCS_DIR.mkdir(exist_ok=True)
    saved_paths = []

    for upload in files:
        if not upload.filename:
            continue

        if not upload.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Apenas arquivos PDF sao permitidos.")

        file_path = DOCS_DIR / upload.filename
        content = await upload.read()
        file_path.write_bytes(content)
        saved_paths.append(str(file_path))

    if not saved_paths:
        raise HTTPException(status_code=400, detail="Nenhum arquivo valido recebido.")

    add_documents_to_store(saved_paths)
    return {"message": "Documentos processados com sucesso.", "files": saved_paths}


@app.post("/materials/generate")
def generate_material(payload: GenerateRequest):
    if not os.path.exists(str(VECTOR_STORE_DIR)):
        raise HTTPException(status_code=400, detail="Banco vetorial nao encontrado. Faca upload antes.")

    try:
        vector_store = load_db()
        response, docs = generate_study_material(
            vector_store=vector_store,
            request_text=payload.request_text,
            output_type=payload.output_type,
            k=payload.k,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    sources = []
    for doc in docs:
        sources.append(
            {
                "preview": doc.page_content[:300],
                "metadata": doc.metadata,
            }
        )

    return {
        "output_type": payload.output_type,
        "response": response,
        "sources": sources,
    }


@app.delete("/database/clear")
def clear_database(delete_docs: bool = Query(True, description="Se true, limpa a pasta doc tambem.")):
    docs_dir = str(DOCS_DIR) if delete_docs else ""
    clear_store_and_docs(docs_dir=docs_dir, vector_store_dir=str(VECTOR_STORE_DIR))
    return {"message": "Limpeza concluida com sucesso."}

@app.get("/v1/models")
def list_models():
    print("🔥 ENTROU NO ENDPOINT")

    return {"teste": "ok"}