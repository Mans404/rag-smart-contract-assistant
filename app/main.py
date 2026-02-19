import os
import tempfile
from fastapi import FastAPI
import gradio as gr

from rag.loader import load_pdf
from rag.chunker import llm_chunk_text
from rag.vectorstore import create_vectorstore
from rag.chains import build_qa_chain, summarize_text

app = FastAPI()

vectorstore = None
qa_chain = None
full_text_cache = ""


def process_pdf(file):

    global vectorstore, qa_chain, full_text_cache

    if file is None:
        return "Please upload a file first."

    file_path = file.name   # ✅ الحل هنا

    documents = load_pdf(file_path)

    full_text = "\n".join([doc.page_content for doc in documents])
    full_text_cache = full_text

    chunks = llm_chunk_text(full_text)

    vectorstore = create_vectorstore(chunks)
    qa_chain = build_qa_chain(vectorstore)

    return "PDF Processed Successfully ✅"



def summarize_pdf():

    if not full_text_cache:
        return "Please process a PDF first."

    return summarize_text(full_text_cache)



def chat_fn(message, history):
    """
    Handle chat messages and return updated history for Gradio Chatbot.
    
    Args:
        message: User's input message (string)
        history: List of previous message pairs
        
    Returns:
        (cleared_message, updated_history): Tuple of empty string and updated history
    """
    
    def _ensure_messages_format(hist):
        """Normalize `hist` into a list of message dicts with 'role' and 'content'.

        Supports incoming formats:
        - list of dicts: [{'role':..., 'content':...}, ...]
        - list of pairs: [[user, bot], [user2, bot2], ...]
        """
        out = []
        if not hist:
            return out
        # Already in dict format
        if isinstance(hist, list) and len(hist) > 0 and isinstance(hist[0], dict) and 'role' in hist[0] and 'content' in hist[0]:
            return hist
        # Convert list of pairs
        for item in hist:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                user, bot = item
                out.append({'role': 'user', 'content': user})
                out.append({'role': 'assistant', 'content': bot})
            elif isinstance(item, dict) and 'role' in item and 'content' in item:
                out.append(item)
            else:
                # Fallback: treat as assistant message string
                out.append({'role': 'assistant', 'content': str(item)})
        return out

    if qa_chain is None:
        history = _ensure_messages_format(history)
        history.append({'role': 'user', 'content': message})
        history.append({'role': 'assistant', 'content': 'Please process a PDF first.'})
        return "", history

    try:
        # Ensure message is a string and not empty
        if not isinstance(message, str) or not message.strip():
            history = _ensure_messages_format(history)
            history.append({'role': 'user', 'content': message})
            history.append({'role': 'assistant', 'content': 'Please enter a valid question.'})
            return "", history
        
        # Call the QA chain with the message
        answer = qa_chain(message.strip())
        
        history = _ensure_messages_format(history)
        history.append({'role': 'user', 'content': message})
        history.append({'role': 'assistant', 'content': answer})
        return "", history
        
    except Exception as e:
        # Handle any errors gracefully
        history = _ensure_messages_format(history)
        error_msg = f"Error: {str(e)}"
        history.append({'role': 'user', 'content': message})
        history.append({'role': 'assistant', 'content': error_msg})
        return "", history


with gr.Blocks() as demo:

    gr.Markdown("# 📄 Smart Contract RAG Assistant")

    with gr.Row():
        pdf_file = gr.File(label="Upload PDF")
        upload_btn = gr.Button("Process PDF")

    status = gr.Textbox(label="Status")

    upload_btn.click(
        process_pdf,
        inputs=pdf_file,
        outputs=status
    )

    gr.Markdown("## 📑 Summarization")
    summarize_btn = gr.Button("Summarize Document")
    summary_output = gr.Textbox(lines=10)

    summarize_btn.click(
        summarize_pdf,
        outputs=summary_output
    )

    gr.Markdown("## 💬 Q&A")
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    msg.submit(chat_fn, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)


app = gr.mount_gradio_app(app, demo, path="/")
