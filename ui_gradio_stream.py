import gradio as gr
import requests
import json

BACKEND_URL = "http://localhost:8000"


def process_pdf(file, session_id_state):
    if file is None:
        return "Please upload a file first.", session_id_state

    with open(file.name, "rb") as f:
        files = {"file": (getattr(file, "orig_name", "upload.pdf"), f, "application/pdf")}
        r = requests.post(f"{BACKEND_URL}/ingest", files=files, timeout=600)

    if r.status_code != 200:
        return f"Error: {r.text}", session_id_state

    data = r.json()
    return data.get("status", "Done"), data.get("session_id")


def summarize_pdf(session_id):
    if not session_id:
        return "Please process a PDF first."

    payload = {"input": {"session_id": session_id}}
    partial = ""

    with requests.post(f"{BACKEND_URL}/summarize/stream", json=payload, stream=True, timeout=600) as r:
        for line in r.iter_lines():
            if line:
                decoded = line.decode("utf-8")
                if decoded.startswith("data:"):
                    data = decoded.replace("data:", "").strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        for op in chunk.get("ops", []):
                            if op.get("op") == "add" and op.get("path") == "/streamed_output/-":
                                token = op.get("value", "")
                                if isinstance(token, str):
                                    partial += token
                                    yield partial
                    except json.JSONDecodeError:
                        continue


def chat_fn(message, history, session_id):
    if not message.strip():
        yield "", history
        return

    if not session_id:
        history = history + [{"role": "user", "content": message}, 
                              {"role": "assistant", "content": "⚠️ Please process a PDF first."}]
        yield "", history
        return

    history = history + [{"role": "user", "content": message},
                         {"role": "assistant", "content": ""}]

    payload = {
        "input": {
            "session_id": session_id,
            "question": message
        }
    }

    with requests.post(f"{BACKEND_URL}/chat/stream", json=payload, stream=True, timeout=600) as r:
        partial_answer = ""
        for line in r.iter_lines():
            if line:
                decoded = line.decode("utf-8")
                if decoded.startswith("data:"):
                    data = decoded.replace("data:", "").strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        for op in chunk.get("ops", []):
                            if op.get("op") == "add" and op.get("path") == "/streamed_output/-":
                                token = op.get("value", "")
                                if isinstance(token, str):
                                    partial_answer += token
                                    history[-1] = {"role": "assistant", "content": partial_answer}
                                    yield "", history
                    except json.JSONDecodeError:
                        continue


with gr.Blocks() as demo:
    gr.Markdown("# 📄 Smart Contract RAG Assistant")

    session_id = gr.State(value="")

    with gr.Row():
        pdf_file = gr.File(label="Upload PDF")
        upload_btn = gr.Button("Process PDF")

    status = gr.Textbox(label="Status")

    upload_btn.click(
        process_pdf,
        inputs=[pdf_file, session_id],
        outputs=[status, session_id]
    )

    gr.Markdown("## 📑 Summarization")
    summarize_btn = gr.Button("Summarize Document")
    summary_output = gr.Textbox(lines=10)

    summarize_btn.click(
        summarize_pdf,
        inputs=[session_id],
        outputs=[summary_output]
    )

    gr.Markdown("## 💬 Q&A")

    chatbot = gr.Chatbot()  #type="messages"  

    msg = gr.Textbox(
        label="Ask a question about the document",
        placeholder="Type anything...",
        lines=4,
        max_lines=10,
        container=True
    )

    with gr.Row():
        send_btn = gr.Button("Enter Your Message ➤", variant="primary", scale=3)
        clear = gr.Button("Clear", scale=1)

    send_btn.click(chat_fn, [msg, chatbot, session_id], [msg, chatbot]).then(
        lambda: "", None, msg
    )
    clear.click(lambda: [], None, chatbot, queue=False)

demo.launch(server_port=7860)