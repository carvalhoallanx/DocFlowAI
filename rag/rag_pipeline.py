import os
import warnings
import requests

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

def load_db():
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        encode_kwargs={"batch_size": 64},
    model_kwargs={'device': 'cpu'}
    )
    # Adicione o parâmetro abaixo para autorizar o carregamento do arquivo
    return FAISS.load_local(
        "vector_store", 
        embeddings, 
        allow_dangerous_deserialization=True
    )


def _call_ollama(prompt):
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
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