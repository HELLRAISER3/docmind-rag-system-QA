import shutil
from pathlib import Path

import gradio as gr

from src.config.configuration import ConfigurationManager
from src.pipeline.ingestion_pipeline import run_ingestion
from src.pipeline.query_pipeline import run_query

cfg = ConfigurationManager()
parse_cfg = cfg.get_parse_config()

DATA_FOLDER = Path(parse_cfg.file_folder)
DATA_FOLDER.mkdir(parents=True, exist_ok=True)


def ingest(files):
    if not files:
        return "No files uploaded!"

    for f in files:
        shutil.copy(f.name, DATA_FOLDER / Path(f.name).name)

    chunks = run_ingestion()
    return (
        f"Ingested **{len(files)}** file(s) → "
        f"**{len(chunks)}** chunks stored in Qdrant!"
    )


def query(user_query: str):
    if not user_query.strip():
        return "Please enter your query!"

    response = run_query(user_query)

    if not response:
        return (
            "No relevant passages found!\n\n"
            "Make sure you have ingested documents first and that your "
            "question relates to their content."
        )

    return response


CSS = """
.title    { text-align: center; font-size: 2rem; font-weight: 700; margin-bottom: 0.2rem; }
.subtitle { text-align: center; color: #6b7280; margin-bottom: 1.5rem; }
"""

with gr.Blocks(title="DocMind RAG", css=CSS, theme=gr.themes.Soft()) as demo:

    gr.Markdown("<div class='title'>DocMind RAG</div>")
    gr.Markdown(
        "<div class='subtitle'>Upload your documents, then ask anything about them.</div>"
    )

    with gr.Tabs():
        with gr.Tab("Upload & Ingest"):
            gr.Markdown(
                "Upload one or more **PDF** files."
            )
            file_input    = gr.File(label="Documents", file_count="multiple", file_types=[".pdf"])
            ingest_btn    = gr.Button("Ingest Documents", variant="primary")
            ingest_status = gr.Markdown(label="Status")

            ingest_btn.click(fn=ingest, inputs=file_input, outputs=ingest_status)

        with gr.Tab("Ask a Question"):
            gr.Markdown(
                "Type your question below. "
                "The answer will be generated from the ingested documents only."
            )
            query_input = gr.Textbox(
                label="Your question",
                placeholder="e.g. What BLEU score did the Transformer achieve on WMT 2014?",
                lines=3,
            )
            query_btn = gr.Button("Get Answer", variant="primary")
            answer    = gr.Markdown(label="Answer")

            query_btn.click(fn=query, inputs=query_input, outputs=answer)
            query_input.submit(fn=query, inputs=query_input, outputs=answer)


if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True)
