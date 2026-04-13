import os
import warnings
import requests
import torch
import ollama
from pathlib import Path
from rag.memory import ChatMemory, load_chat
from rag.insights import load_insights

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"
memory = ChatMemory(max_history=10)

# Configuração do modelo Ollama - pode ser alterada via variável de ambiente
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")

def load_db():
    from langchain_community.vectorstores import FAISS
    from langchain_ollama import OllamaEmbeddings

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store_path = Path(__file__).resolve().parents[1] / "vector_store"
    
    # Verificar se o diretório e arquivos existem
    if not vector_store_path.exists():
        raise FileNotFoundError(f"Banco de dados não encontrado em {vector_store_path}. Faça upload de um documento primeiro.")
    
    try:
        return FAISS.load_local(
            str(vector_store_path), 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar banco de dados FAISS: {e}")


def _call_ollama(prompt):
    model = os.getenv("OLLAMA_MODEL", "gemma3:4b")
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

    try:
        response = requests.post(
            ollama_url,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        answer = data.get("response", "").strip()
        if not answer:
            answer = "Nao consegui gerar uma resposta com o Ollama."
    except requests.RequestException as exc:
        raise RuntimeError(
            "Falha ao conectar no Ollama. Verifique se o servidor esta ativo em "
            "http://localhost:11434 e se o modelo foi baixado."
        ) from exc

    return answer


def generate_study_material(vector_store, request_text, output_type="resposta", k=3):
    docs = vector_store.similarity_search(request_text, k=k)
    context = "\n".join([doc.page_content for doc in docs])

    instructions_by_type = {
        "resposta": (
            "Responda objetivamente ao pedido do usuario com base no contexto. "
            "Se faltar informacao, diga explicitamente."
        ),
        "anotacoes": (
            "Crie anotacoes de pesquisa em formato de topicos. "
            "Inclua: conceitos-chave, definicoes, exemplos e pontos para aprofundar."
        ),
        "resumo": (
            "Crie um resumo estruturado com: visao geral, principais ideias, "
            "conclusoes e termos importantes."
        ),
        "mapa_mental": (
            "Monte um mapa mental textual em arvore usando Markdown com niveis "
            "de hierarquia claros (raiz -> ramos -> sub-ramos)."
        ),
        "cronograma": (
            "Crie um cronograma de estudo em semanas baseado no conteudo, "
            "com objetivos, atividades e entregaveis por semana."
        ),
        "questionario": (
            "Crie um questionario com 10 perguntas: 6 objetivas e 4 discursivas. "
            "No final, inclua gabarito comentado."
        ),
    }

    instruction = instructions_by_type.get(output_type, instructions_by_type["resposta"])
    prompt = f"""Voce e um assistente academico.
Use somente o contexto abaixo para gerar a saida.
Se nao houver base no contexto, informe claramente.

Tipo de saida: {output_type}
Instrucao de formato: {instruction}

Contexto:
{context}

Pedido do usuario:
{request_text}
"""

    answer = _call_ollama(prompt)
    return answer, docs

def answer_question(vector_store, query, k=3):
    return generate_study_material(
        vector_store=vector_store,
        request_text=query,
        output_type="resposta",
        k=k
    )

def generate_response(vector_store, question, k=3):
    docs = vector_store.similarity_search(question, k=k)

    context = "\n\n".join([d.page_content for d in docs])
    insights = load_insights()
    history = memory.get_context()

    prompt = f"""
    Você é um assistente inteligente.

    HISTÓRICO:
    {history}

    INSIGHTS:
    {insights}

    CONTEXTO:
    {context}

    PERGUNTA:
    {question}
    """

    stream = ollama.chat(
        model="gemma3:4b",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    full_response = ""

    for chunk in stream:
        content = chunk["message"]["content"]
        full_response += content
        yield content, docs

    memory.add("user", question)
    memory.add("assistant", full_response)

def ask_llm(question, db, chat_id):
    history = load_chat(chat_id)

    docs = db.similarity_search(question, k=4)

    context = "\n\n".join([d.page_content for d in docs])

    messages = history + [
        {
            "role": "user",
            "content": f"""
            Contexto:
            {context}

            Pergunta:
            {question}
            """
        }
    ]

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        stream=True
    )

    return response, docs