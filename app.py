import ast
import html
import os
from pathlib import Path
import arxiv
from urllib.parse import urlparse

import faiss
import gradio as gr
import numpy as np
import pandas as pd
from sqlalchemy import bindparam, create_engine, text
from sqlalchemy.engine import URL

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from dotenv import load_dotenv
load_dotenv()


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_INDEX_PATH = DATA_DIR / "BERT_embeddings" / "faiss_index.index"

MODEL_NAME = "sentence-transformers/all-distilroberta-v1" # see https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
RESULT_COLUMNS = ["id", "paper_id", "title", "authors", "categories", "update_year"]



# can be either "cpu", "cuda", or "mps" (for Apple Silicon)
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
def build_engine():
    url = URL.create(
        drivername="mysql+pymysql",
        username=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT")),
        database=os.getenv("DB_NAME"),
    )
    return create_engine(url, pool_pre_ping=True)


# the index is stored in HF Bucket
def load_index():
    from huggingface_hub import download_bucket_files
    OUTPUT_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    download_bucket_files(
        bucket_id="salirafi/AstRoBERT_index",
        files=[
            ("data/BERT_embeddings/faiss_index.index", str(OUTPUT_INDEX_PATH)),
        ],
    )

    return faiss.read_index(str(OUTPUT_INDEX_PATH))


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()    
    return tokenizer, model


ENGINE = build_engine()
DEVICE = get_device()
INDEX = load_index()
TOKENIZER, MODEL = load_model()


def embed(text_input: str) -> np.ndarray:
    encoded = TOKENIZER(
        [text_input],
        padding=True,
        truncation=True,
        max_length=512, # max_length for distilroberta
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        output = MODEL(**encoded)

    token_embeddings = output.last_hidden_state
    mask_expanded = encoded.attention_mask.unsqueeze(-1).float()
    sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
    sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
    mean_embeddings = sum_embeddings / sum_mask
    normalized = F.normalize(mean_embeddings, p=2, dim=1)
    return normalized.cpu().numpy().astype("float32")



# parse categories column into readable format
def format_categories(raw_value: str) -> str:
    if not raw_value:
        return ""

    try:
        parsed = ast.literal_eval(raw_value)
    except (ValueError, SyntaxError):
        return str(raw_value)

    if isinstance(parsed, list):
        return ", ".join(str(item) for item in parsed)
    return str(parsed)


def fetch_recommendation_rows(ids: list[int]) -> pd.DataFrame:
    if not ids:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    stmt = (
        text(
            """
            SELECT id, paper_id, title, authors, categories, update_year
            FROM papers
            WHERE id IN :ids
            """
        )
        .bindparams(bindparam("ids", expanding=True))
    )

    with ENGINE.connect() as connection:
        rows = pd.read_sql_query(stmt, connection, params={"ids": ids})

    order_map = {paper_id: position for position, paper_id in enumerate(ids)}
    rows["rank"] = rows["id"].map(order_map) # add a temporary column to preserve the original order of paper_ids
    rows = rows.sort_values("rank").drop(columns=["rank"]) # sort by the original order and remove the temporary column
    return rows


def format_authors(raw_value: str) -> str:
    if not raw_value:
        return "Unknown authors"

    try:
        parsed = ast.literal_eval(raw_value) # authors stored as stringified lists, parse them back into Python lists for display
    except (ValueError, SyntaxError):
        return str(raw_value)

    if isinstance(parsed, list):
        return ", ".join(str(author) for author in parsed)

    return str(parsed)

# UI helper for display results
def build_results_html(rows: pd.DataFrame) -> str:
    if rows.empty:
        return '<div class="result-empty">No recommendations found.</div>'


    cards: list[str] = []
    for _, row in rows.iterrows():
        title = html.escape(str(row.get("title", "") or "Untitled"))
        authors = html.escape(format_authors(row.get("authors", ""))) # parsed authors' list
        year = html.escape(str(row.get("update_year", "") or "Unknown year"))
        paper_id = html.escape(str(row.get("paper_id", "") or "").strip())
        # Relevance = html.escape(str(row.get("Relevance", "") or ""))

        if paper_id:
            paper_url = f"https://arxiv.org/abs/{paper_id}"
            link_html = (
                f'<a href="{paper_url}" target="_blank" rel="noopener noreferrer">{paper_url}</a>'
            )
        else:
            link_html = "<span>No paper link available.</span>"

        cards.append(
            f"""
            <article class="result-card">
                <div class="result-title">{title}</div>
                <div class="result-meta">{year}</div>
                <div class="result-meta">{authors}</div>
                <div class="result-link">{link_html}</div>
            </article>
            """
        )

    return f'<div class="results-list">{"".join(cards)}</div>'



def get_abstract_from_url(url: str) -> str:
    path = urlparse(url).path.strip("/")

    if path.startswith("abs/"):
        paper_id = path.removeprefix("abs/")
        client = arxiv.Client() # arXiv's API wrapper
        paper = next(client.results(arxiv.Search(id_list=[paper_id])))
        return paper.summary # the abstract

    return ""
    


def recommend(abstract: str, top_k: int = 10, sort_by: str = "Relevance") -> str:
    cleaned = abstract.strip()

    if "arxiv.org/abs/" in abstract: # if the input looks like an arXiv URL, extract the paper ID and fetch its abstract
        cleaned = get_abstract_from_url(abstract)

    if not cleaned:
        return '<div class="result-empty">Please enter an abstract or valid arXiv URL.</div>'

    search_k = min(top_k + 5, INDEX.ntotal)
    query_vector = embed(cleaned)
    scores, indices = INDEX.search(query_vector, search_k) # return paper IDs directly from index

    ranked_ids: list[int] = []
    scores_by_id: dict[int, float] = {}
    for idx, score in zip(indices[0], scores[0]):
        if idx < 0:
            continue
        if score >= 0.9999:
            continue
        ranked_ids.append(idx)
        scores_by_id[idx] = score
        if len(ranked_ids) >= top_k:
            break

    rows = fetch_recommendation_rows(ranked_ids)


    if sort_by == "Relevance":
        rows["Relevance"] = rows["id"].map(scores_by_id)
        rows = rows.sort_values("Relevance", ascending=False)
    elif sort_by == "Year":
        rows["year_sort"] = pd.to_numeric(rows["update_year"], errors="coerce")
        rows = rows.sort_values("year_sort", ascending=False, na_position="last")

    return build_results_html(rows)


with gr.Blocks(
    title="Paper Recommender",
    theme=gr.themes.Ocean(),
    css="""
    .app-shell {
        max-width: 1100px;
        margin: 0 auto;
        padding: 24px 16px 32px;
    }
    .app-title h1 {
        font-size: 2.4rem;
        line-height: 1.15;
        margin-bottom: 0.9rem;
    }
    .app-title p {
        font-size: 1.05rem;
        line-height: 1.4;
        color: #374151;
        margin: 0 0 0.7rem 0;
    }

    .external-link {
        color: #2563eb;
        text-decoration: none;
    }

    .results-list {
        margin-top: 1.25rem;
        display: flex;
        flex-direction: column;
        gap: 0.9rem;
    }


    .result-card {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        background: #ffffff;
        color: #000000;
        padding: 1rem 1.1rem;
    }
    .result-title {
        font-size: 1.1rem;
        font-weight: 700;
        line-height: 1.4;
        color: #111827;
        margin-bottom: 0.35rem;}
    .result-meta {
        font-size: 0.95rem;
        color: #4b5563;
        margin-bottom: 0.55rem;
    }

    .result-link a,
    .result-link span {
        font-size: 0.95rem;
        word-break: break-word;
    }


    .result-empty {
        margin-top: 1.25rem;
        padding: 1rem 1.1rem;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        background: #f9fafb;
        color: #374151;
    }
    """,
) as demo:
    with gr.Column(elem_classes="app-shell"):
        gr.Markdown(
            """
            <div class="app-title">
                <h1>AstRoBERT</h1>
                <p>
                    Retrieving semantically similar astrophysical papers from ArXiv. Includes ~381,000 papers uploaded per March 2026.
                </p>
                <p>
                    Powered by <a href="https://huggingface.co/sentence-transformers/all-distilroberta-v1" target="_blank" class="external-link">distilRoBERTa transformer</a> 
                    and <a href="https://faiss.ai/" target="_blank" class="external-link">FAISS</a>.
                </p>
            </div>
            """
        )

        abstract_input = gr.Textbox(
            lines=8,
            placeholder="Paste an abstract or https://arxiv.org/abs/<paper_id> here...",
            label="Abstract or arXiv URL",
        )
        top_k_input = gr.Slider(
            minimum=1,
            maximum=30,
            value=10,
            step=1,
            label="Number of results",
        )
        sort_input = gr.Dropdown(
            choices=["Relevance", "Year"],
            value="Relevance",
            label="Sort results by",
        )
        submit_button = gr.Button("Get Recommendations", variant="primary")
        results_output = gr.HTML(label="Recommended Papers")



        submit_button.click(
            fn=recommend,
            inputs=[abstract_input, top_k_input, sort_input],
            outputs=results_output,
        )
        abstract_input.submit(
            fn=recommend,
            inputs=[abstract_input, top_k_input, sort_input],
            outputs=results_output,
        )

demo.launch(share=False)
