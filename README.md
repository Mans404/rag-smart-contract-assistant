# RAG Smart Contract Assistant

## Backend
LangServe API for:
- PDF Processing
- Summarization
- Q&A

## Frontend
Gradio UI

## Setup

1. Install dependencies:
pip install -r requirements.txt

2. Create .env file with:
OPENAI_API_KEY=your_key
MODEL_NAME=openai/gpt-4o-mini
OPENAI_BASE_URL=https://openrouter.ai/api/v1
TOP_K=5

3. Run backend:
python rag/serve.py

4. Run frontend:
python app/gradio_frontend.py
