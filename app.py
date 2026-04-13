from pathlib import Path
import os
import uuid
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
import streamlit as st

from rag.rag_pipeline import generate_study_material, load_db
from rag.ingest import add_documents_to_store, clear_store_and_docs
from rag.memory import save_message, load_chat

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="DocFlow AI", layout="wide")

css = (BASE_DIR / "styles.css").read_text(encoding="utf-8")
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# =============================
# SESSION STATE
# =============================
if "chat_id" not in st.session_state:
    st.session_state.chat_id = str(uuid.uuid4())

# 🧠 Estado da conversa
if "messages" not in st.session_state:
    st.session_state.messages = load_chat(st.session_state.chat_id)
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "upload_done" not in st.session_state:
    st.session_state.upload_done = False
if "clear_done" not in st.session_state:
    st.session_state.clear_done = False

# Carregar banco vetorial
try:
    db = load_db()
    has_vector_store = True
except Exception as e:
    has_vector_store = False
    db = None
    # print(f"Aviso: Banco de dados não carregado: {e}")

# 🎨 Sidebar
with st.sidebar:
    st.title("💬 Gerenciador de Chat")
    
    if st.button("➕ Novo Chat"):
        st.session_state.chat_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
    
    st.text(f"Chat ID: {st.session_state.chat_id[:8]}...")
    
    st.markdown("---")
    st.title("📄 Upload de Documento")

    uploaded_files = st.file_uploader(
        "Envie PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"pdf_uploader_{st.session_state.uploader_key}"
    )

    if uploaded_files:
        docs_dir = str(BASE_DIR / "doc")
        os.makedirs(docs_dir, exist_ok=True)
        paths = []
        for file in uploaded_files:
            path = os.path.join(docs_dir, file.name)
            with open(path, "wb") as f:
                f.write(file.read())
            paths.append(path)

        with st.spinner("Processando documento..."):
            add_documents_to_store(paths)

        st.session_state.upload_done = True
        st.session_state.uploader_key += 1
        st.rerun()
    if st.session_state.upload_done:
        st.success("✅ PDFs adicionados!")
        st.session_state.upload_done = False
    if st.session_state.clear_done:
        st.success("🧹 Banco e documentos removidos com sucesso!")
        st.session_state.clear_done = False
    st.markdown("---")
    st.markdown("### ⚙️ Configurações")
    k = st.slider("Quantidade de resultados (k)", 1, 10, 3)
    output_type = st.selectbox(
        "Tipo de material gerado",
        options=[
            "resposta",
            "anotacoes",
            "resumo",
            "mapa_mental",
            "cronograma",
            "questionario",
        ],
        index=0,
    )
    if st.button("🗑️ Limpar chat"):
        st.session_state.messages = []
        st.rerun()
    if st.button("🧹 Limpar banco"):
        clear_store_and_docs()
        st.session_state.messages = []
        st.session_state.uploader_key += 1
        st.session_state.clear_done = True
        st.rerun()

# 🧾 Título
st.title("🤖 ChatDOC")

# 💬 Exibir histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if not has_vector_store:
    st.warning("⚠️ Envie um documento primeiro para usar o chat")

# 🧠 Input do usuário
if prompt := st.chat_input("Digite sua pergunta..."):
    if has_vector_store:
        # Salva pergunta
        save_message(st.session_state.chat_id, "user", prompt)
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        # Mostra pergunta
        with st.chat_message("user"):
            st.markdown(prompt)

        # Gera resposta
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                try:
                    # Usa generate_study_material com output_type selecionado
                    vector_store = load_db()
                    response, docs = generate_study_material(
                        vector_store=vector_store,
                        request_text=prompt,
                        output_type=output_type,
                        k=k
                    )

                    st.markdown(response)

                    # Salva resposta
                    save_message(st.session_state.chat_id, "assistant", response)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })

                    # Mostrar fontes
                    with st.expander("📚 Fontes utilizadas"):
                        if docs:
                            for i, doc in enumerate(docs, 1):
                                st.markdown(f"**Fonte {i}:**")
                                st.write(doc.page_content[:300] + "...")
                                st.write(doc.metadata)
                        else:
                            st.info("Nenhuma fonte encontrada")

                except Exception as e:
                    st.error(f"❌ Erro: {e}")
    else:
        st.warning("⚠️ Envie um documento primeiro")