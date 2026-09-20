import gradio as gr

from src.pipeline import ingest_pdfs, ingest_url
from src.rag_pipeline import answer_question, refresh_retriever
from src.workspace_manager import WorkspaceManager

import os
import time
import uuid


SESSION_WORKSPACE = f"session_{uuid.uuid4().hex[:8]}"
WorkspaceManager.set_workspace(SESSION_WORKSPACE)
print(f"Session workspace: {SESSION_WORKSPACE}")
UPLOADED_FILES = []


examples = [
    "What are attention mechanisms?",
    "What are transformers?",
    "Explain generative AI",
    "What are LLMs?",
]


def create_session_workspace():
    return f"session_{uuid.uuid4().hex[:8]}"


def update_workspace(name):
    WorkspaceManager.set_workspace(name)
    workspace = WorkspaceManager.get_workspace()
    return f"Active workspace: {workspace}"


def upload_pdf(files, workspace_name):
    WorkspaceManager.set_workspace(workspace_name)

    if not files:
        return "\n".join(UPLOADED_FILES), "No files uploaded."

    paths = []

    for file in files:
        if file.name not in UPLOADED_FILES:
            UPLOADED_FILES.append(file.name)

    paths.extend(UPLOADED_FILES)

    try:
        chunks = ingest_pdfs(paths)
        workspace = WorkspaceManager.get_workspace()
        refresh_retriever(workspace_name=workspace)
    except Exception as exc:
        return "\n".join(UPLOADED_FILES), f"PDF ingestion failed:\n{exc}"

    return "\n".join(UPLOADED_FILES), (
        f"Successfully ingested {len(files)} PDFs\n"
        f"Workspace: {workspace}\n"
        f"Uploaded files: {len(UPLOADED_FILES)}\n"
        f"Chunks created: {chunks}"
    )


def ingest_webpage(url, workspace_name):
    WorkspaceManager.set_workspace(workspace_name)
    workspace = WorkspaceManager.get_workspace()

    if not url or not url.strip():
        return "Please provide a valid URL."

    try:
        result = ingest_url(
            url.strip(),
        )
        refresh_retriever(workspace_name=workspace)
    except Exception as exc:
        return f"URL ingestion failed:\n{exc}"

    return (
        f"Workspace: {workspace}\n"
        f"Successfully ingested:\n"
        f"{result['title']}"
    )


def _ask_rag(question, workspace):
    result = answer_question(
        question,
        workspace_name=workspace,
    )

    answer = result["answer"]
    sources = result.get("sources", [])

    source_text = ""
    for idx, src in enumerate(sources, start=1):
        source_type = "[WEB]" if src.get("type") == "webpage" else "[PDF]"
        metadata = src.get("metadata", {})
        page = metadata.get("page", src.get("page", "?"))
        source = src["source"]
        if src.get("type") != "webpage":
            source = f"{source} (Page {page})"
        source_text += (
            f"[{idx}] {source_type} {source}\n"
            f"{src['preview']}\n\n"
        )

    return answer, source_text


def chat(message, history, workspace_name=SESSION_WORKSPACE):
    WorkspaceManager.set_workspace(workspace_name)

    result = answer_question(
        message,
        workspace_name=workspace_name,
    )

    return result["answer"]


def submit_message(message, history, workspace_name):
    WorkspaceManager.set_workspace(workspace_name)

    if not message or not message.strip():
        return history, "", "Enter a question to search your knowledge base."

    response, sources = _ask_rag(
        message.strip(),
        WorkspaceManager.get_workspace()
    )
    updated_history = history + [
        {"role": "user", "content": message.strip()},
        {"role": "assistant", "content": response},
    ]

    sources = sources.strip() if sources else "No retrieved sources returned."
    return updated_history, "", sources


def clear_chat():
    return [], ""


def reset_session():
    global SESSION_WORKSPACE

    UPLOADED_FILES.clear()
    SESSION_WORKSPACE = create_session_workspace()
    WorkspaceManager.set_workspace(SESSION_WORKSPACE)

    return (
        SESSION_WORKSPACE,
        "",
        f"Active workspace: {SESSION_WORKSPACE}",
    )


theme = gr.themes.Soft(
    primary_hue="blue",
    neutral_hue="slate",
    radius_size="md",
).set(
    body_background_fill="#f8fafc",
    block_background_fill="#ffffff",
    block_border_width="1px",
    block_shadow="0 10px 30px rgba(15, 23, 42, 0.06)",
    button_primary_background_fill="#2563eb",
    button_primary_background_fill_hover="#1d4ed8",
)

css = """
.gradio-container {
    max-width: 1280px !important;
    margin: 0 auto;
}
.app-shell {
    gap: 18px;
}
.sidebar {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 18px;
}
.main-panel {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 18px;
}
.system-info {
    color: #475569;
    font-size: 13px;
    line-height: 1.55;
}
footer {
    display: none !important;
}
"""


with gr.Blocks(theme=theme, css=css, title="RAG Document Analyzer", analytics_enabled=False) as demo:
    gr.Markdown(
        "# RAG Document Analyzer\n"
        "Hybrid Retrieval + Reranking + Evaluation"
    )

    with gr.Row(elem_classes=["app-shell"]):
        with gr.Column(scale=1, min_width=280, elem_classes=["sidebar"]):
            gr.Markdown("### Document Workspace")
            workspace_box = gr.Textbox(
                label="Workspace Name",
                value=SESSION_WORKSPACE
            )
            pdf_upload = gr.File(
                label="Upload PDFs",
                file_count="multiple",
                file_types=[".pdf"],
                height=160,
            )
            uploaded_files_box = gr.Textbox(
                label="Uploaded Documents",
                lines=10,
                interactive=False
            )
            status_box = gr.Textbox(
                label="Status",
                lines=5,
                interactive=False,
                placeholder="Uploads and URL ingestion status will appear here.",
            )

            url_input = gr.Textbox(
                label="Add Webpage URL",
                placeholder="https://example.com"
            )
            url_button = gr.Button("Ingest URL")

            clear_button = gr.Button("Clear chat", variant="secondary")
            reset_session_button = gr.Button("Reset Session", variant="secondary")

            gr.Markdown("### System Info")
            gr.Markdown(
                "<div class='system-info'>"
                "<strong>Retrieval:</strong> Hybrid search<br>"
                "<strong>Generation:</strong> Groq-backed LLM<br>"
                "<strong>Sources:</strong> Returned context preview<br>"
                "<strong>Status:</strong> PDF upload placeholder and URL ingestion enabled"
                "</div>"
            )

        with gr.Column(scale=3, elem_classes=["main-panel"]):
            chatbot = gr.Chatbot(
                label="Ask your documents",
                height=520,
                buttons=["copy", "copy_all"],
                avatar_images=(None, None),
                layout="bubble",
            )

            with gr.Row():
                message_box = gr.Textbox(
                    label="Question",
                    placeholder="Ask a question about your documents...",
                    lines=2,
                    scale=8,
                    autofocus=True,
                )
                submit_button = gr.Button("Submit", variant="primary", scale=1)

            gr.Examples(
                examples=examples,
                inputs=message_box,
                label="Example Questions",
            )

            gr.Markdown("### Sources")
            sources_box = gr.Textbox(
                label="Retrieved Sources",
                lines=12,
                interactive=False,
                placeholder="Retrieved context will appear after each query.",
            )

    pdf_upload.upload(
        fn=upload_pdf,
        inputs=[pdf_upload, workspace_box],
        outputs=[uploaded_files_box, status_box],
    )
    workspace_box.change(
        update_workspace,
        inputs=workspace_box,
        outputs=status_box,
    )
    url_button.click(
        ingest_webpage,
        inputs=[url_input, workspace_box],
        outputs=status_box,
    )
    submit_button.click(
        fn=submit_message,
        inputs=[message_box, chatbot, workspace_box],
        outputs=[chatbot, message_box, sources_box],
    )
    message_box.submit(
        fn=submit_message,
        inputs=[message_box, chatbot, workspace_box],
        outputs=[chatbot, message_box, sources_box],
    )
    clear_button.click(
        fn=clear_chat,
        inputs=None,
        outputs=[chatbot, sources_box],
    )
    reset_session_button.click(
        fn=reset_session,
        inputs=None,
        outputs=[workspace_box, uploaded_files_box, status_box],
    )


if __name__ == "__main__":
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    demo.queue()
    demo.launch()
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
