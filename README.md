# 🧠 RAG Smart Contract Assistant

A Retrieval-Augmented Generation (RAG) system for:

- 📄 Uploading and processing PDF documents
- 📑 Summarizing documents
- 💬 Asking questions with chat history
- 🔎 Semantic search using FAISS
- 🚀 Backend powered by LangServe
- 🎨 Frontend powered by Gradio

---

# 📁 Project Structure

rag_project/
│
├── app/
│ └── langserve_backend.py # LangServe backend API
│
├── rag/
│ ├── init.py
│ ├── loader.py # PDF loading
│ ├── chunker.py # LLM-based chunking
│ ├── embeddings.py # Embeddings
│ ├── vectorstore.py # FAISS creation
│ ├── chains.py # QA & Summarization chains
│ └── config.py # Environment configuration
│
├── .env # API Keys (NOT pushed to GitHub)
├── .gitignore
├── ui_gradio_stream.py # Gradio UI (Frontend)
├── prompt.py
├── requirements.txt
└── README.md


---

# ⚙️ Setup Instructions

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/rag-smart-contract-assistant.git
cd rag-smart-contract-assistant
```
## 2️⃣ Create Virtual Environment
### python -m venv .venv
### Activate it in Windows:
#### .venv\Scripts\activate
## 3️⃣ Install Dependencies
#### pip install -r requirements.txt
## pip install -r requirements.txt
## 4️⃣ Create a file named .env in the root directory and add:
```bash
OPENAI_API_KEY=your_openrouter_key_here
OPENAI_BASE_URL=https://openrouter.ai/api/v1
MODEL_NAME=openai/gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
TOP_K=5
```
# 🚀 Running the Application
## 🔹 Step 1: Start Backend (LangServe)
### python rag/serve.py
## 🔹 Step 2: Start Frontend (Gradio)
## Open a new terminal and run:
### python app/gradio_frontend.py

# 🧪 How to Use

## Upload a PDF

## Click "Process PDF"

## Click "Summarize Document" to get summary

## Ask questions in chatbox

## 🔎 Backend API Endpoints

| Method | Endpoint    | Description  |
| ------ | ----------- | ------------ |
| POST   | /upload_pdf | Process PDF  |
| GET    | /summarize  | Get summary  |
| POST   | /ask        | Ask question |


# 🧠 Technologies Used

## LangChain

## FAISS

## OpenRouter (gpt-4o-mini)

## LangServe

## FastAPI

## Gradio
