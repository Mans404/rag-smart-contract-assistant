from __future__ import annotations

import os
import uuid
import tempfile
from typing import Dict, Any, Iterator

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from langserve import add_routes
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, Runnable

from rag.loader import load_pdf
from rag.chunker import llm_chunk_text
from rag.vectorstore import create_vectorstore
from rag.chains import get_llm
from rag.config import TOP_K


SESSIONS: Dict[str, Dict[str, Any]] = {}

QA_PROMPT = ChatPromptTemplate.from_template(
    """
    Answer the question using ONLY the provided context.

    Context:
    {context}

    Question:
    {question}
    """
)

SUM_PROMPT = ChatPromptTemplate.from_template(
    """
    Summarize this document clearly:

    {text}
    """
)


# ========= Custom Runnable يدعم stream =========
class ChatInput(BaseModel):
    session_id: str
    question: str

class SummarizeInput(BaseModel):
    session_id: str


class ChatRunnable(Runnable):
    def invoke(self, input: ChatInput, config: RunnableConfig = None) -> str:
        return "".join(self.stream(input, config))

    def stream(self, input: ChatInput, config: RunnableConfig = None, **kwargs) -> Iterator[str]:
        if isinstance(input, dict):
            input = ChatInput(**input)

        session = SESSIONS.get(input.session_id)
        if session is None:
            yield "Please process a PDF first (invalid session)."
            return

        question = (input.question or "").strip()
        if not question:
            yield "Please enter a valid question."
            return

        retriever = session["retriever"]
        docs = retriever.invoke(question)
        context = "\n\n".join([d.page_content for d in docs])

        llm = get_llm()
        chain = QA_PROMPT | llm

        for chunk in chain.stream({"context": context, "question": question}):
            yield chunk.content


class SummarizeRunnable(Runnable):
    def invoke(self, input: SummarizeInput, config: RunnableConfig = None) -> str:
        return "".join(self.stream(input, config))

    def stream(self, input: SummarizeInput, config: RunnableConfig = None, **kwargs) -> Iterator[str]:
        if isinstance(input, dict):
            input = SummarizeInput(**input)

        session = SESSIONS.get(input.session_id)
        if session is None:
            yield "Please process a PDF first (invalid session)."
            return

        full_text = session.get("full_text") or ""
        if not full_text:
            yield "Please process a PDF first."
            return

        llm = get_llm()
        chain = SUM_PROMPT | llm

        for chunk in chain.stream({"text": full_text}):
            yield chunk.content


# ========= App =========
app = FastAPI(title="RAG Backend (LangServe)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    suffix = os.path.splitext(file.filename or "")[1] or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        tmp.write(await file.read())

    try:
        documents = load_pdf(tmp_path)
        full_text = "\n".join([doc.page_content for doc in documents])

        chunks = llm_chunk_text(full_text)
        vectorstore = create_vectorstore(chunks)
        retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

        session_id = str(uuid.uuid4())
        SESSIONS[session_id] = {"retriever": retriever, "full_text": full_text}

        return {"status": "PDF Processed Successfully ✅", "session_id": session_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


add_routes(app, ChatRunnable(), path="/chat", input_type=ChatInput)
add_routes(app, SummarizeRunnable(), path="/summarize", input_type=SummarizeInput)


@app.get("/health")
def health():
    return {"ok": True, "sessions": len(SESSIONS)}