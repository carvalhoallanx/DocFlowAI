📄🤖 Chat com Documentos (RAG em Python)

Um assistente inteligente que permite conversar com seus próprios arquivos usando IA.

🧠 Sobre o projeto

Este projeto implementa um sistema de RAG (Retrieval-Augmented Generation) que permite:

📄 Fazer upload de documentos (PDF, TXT, etc)
❓ Fazer perguntas sobre o conteúdo
🤖 Receber respostas baseadas nos dados enviados

👉 Basicamente: seu próprio ChatGPT com arquivos personalizados

⚙️ Tecnologias utilizadas
🐍 Python
🔗 LangChain
💬 Ollama
🤖 StreamLit
🧠 Embeddings (Sentence Transformers)
📊 FAISS (Vector Store)
🚀 FastAPI

## API FastAPI

Executar API:

`uvicorn api.main:app --reload`

Endpoints principais:

- `GET /health`
- `POST /documents/upload` (multipart/form-data com arquivos PDF)
- `POST /materials/generate` (JSON com `request_text`, `output_type`, `k`)
- `DELETE /database/clear?delete_docs=true`
