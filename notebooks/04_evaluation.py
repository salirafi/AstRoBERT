#!usr/bin/env python3

# script to run the evaluation code from 04_evaluation.ipynb
# i can't figure out why the notebook keeps crashing when running recommend() function so I switch to .py file
# it should work fine

from pathlib import Path
import ast

import faiss
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ABSTRACTS_PATH = DATA_DIR / "arxiv_metadata.csv"
TFIDF_VECTORIZER_PATH = DATA_DIR / "tfidf_vectorizer.joblib"
TFIDF_MATRIX_PATH = DATA_DIR / "tfidf_matrix.joblib"
OUTPUT_INDEX_PATH = DATA_DIR / "BERT_embeddings" / "faiss_index.index"

MODEL_NAME = "sentence-transformers/all-distilroberta-v1"
TOP_K = 10
EVAL_TOP_K = 5
SAMPLE_SIZE = 1000
RUN_TFIDF_BASELINE = True



def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_df() -> pd.DataFrame:
    df = pd.read_csv(ABSTRACTS_PATH)
    return df.sort_values("id").reset_index(drop=True)
def load_index():
    return faiss.read_index(str(OUTPUT_INDEX_PATH))


def load_model(device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    return tokenizer, model


# make sure to run with the same config as embedding
def embed(text_input: str, tokenizer, model, device: torch.device) -> np.ndarray:
    encoded = tokenizer(
        [text_input],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output = model(**encoded)

    token_embeddings = output.last_hidden_state
    mask_expanded = encoded.attention_mask.unsqueeze(-1).float()
    sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
    sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
    mean_embeddings = sum_embeddings / sum_mask
    normalized = F.normalize(mean_embeddings, p=2, dim=1)
    return normalized.cpu().numpy().astype("float32")



def parse_categories(raw_value: str) -> set[str]:
    if pd.isna(raw_value) or raw_value == "":
        return set()

    try:
        parsed = ast.literal_eval(raw_value)
    except (ValueError, SyntaxError):
        return {str(raw_value)}

    if isinstance(parsed, list):
        return {str(item) for item in parsed}
    return {str(parsed)}

def format_categories(raw_value: str) -> str:
    return ", ".join(sorted(parse_categories(raw_value))) # for display


# this is to ensure to return results in the same order as the input ids
def fetch_rows_by_ids(df: pd.DataFrame, ids: list[int]) -> pd.DataFrame:
    if not ids:
        return pd.DataFrame(columns=df.columns)
    rows = df[df["id"].isin(ids)].copy()
    order_map = {paper_id: position for position, paper_id in enumerate(ids)}
    rows["rank"] = rows["id"].map(order_map)
    return rows.sort_values("rank").drop(columns=["rank"])



def recommend(
    text_input: str,
    df: pd.DataFrame,
    index,
    tokenizer,
    model,
    device: torch.device,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    cleaned = text_input.strip()
    if not cleaned:
        return pd.DataFrame(columns=["paper_id", "title", "categories", "update_year", "score"])

    search_k = min(top_k + 5, index.ntotal)
    query_vector = embed(cleaned, tokenizer, model, device)
    scores, indices = index.search(query_vector, search_k)

    ranked_ids: list[int] = []
    scores_by_id: dict[int, float] = {}
    for idx, score in zip(indices[0], scores[0]):

        # skip invalid indices and self-matches
        if idx < 0:
            continue
        if score >= 0.9999:
            continue

        ranked_ids.append(int(idx))
        scores_by_id[int(idx)] = float(score)
        if len(ranked_ids) >= top_k:
            break

    rows = fetch_rows_by_ids(df, ranked_ids)
    if rows.empty:
        return pd.DataFrame(columns=["paper_id", "title", "categories", "update_year", "score"])

    rows["score"] = rows["id"].map(scores_by_id)
    rows["categories"] = rows["categories"].map(format_categories)
    return rows[["paper_id", "title", "categories", "update_year", "score"]]

def precision_at_k(
    query_idx: int,
    df: pd.DataFrame,
    index,
    tokenizer,
    model,
    device: torch.device,
    top_k: int = EVAL_TOP_K,
) -> float:
    query_row = df.iloc[query_idx]
    query_abstract = str(query_row["abstract"]).strip()
    query_id = int(query_row["id"])
    query_cats = parse_categories(query_row["categories"])

    query_vector = embed(query_abstract, tokenizer, model, device)
    scores, indices = index.search(query_vector, top_k + 10)
    retrieved_ids = [
        int(idx)
        for idx in indices[0]
        if idx >= 0 and int(idx) != query_id # exclude self-match
    ][:top_k]

    if not retrieved_ids:
        return 0.0


    retrieved_rows = fetch_rows_by_ids(df, retrieved_ids)
    matches = sum(
        1
        for raw_value in retrieved_rows["categories"]
        if parse_categories(raw_value) & query_cats
    )
    return matches / top_k # precision@k where matches are defined by any shared category


def load_tfidf_matrix():
    vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
    tfidf_matrix = joblib.load(TFIDF_MATRIX_PATH)
    return vectorizer, tfidf_matrix

def precision_at_k_tfidf(
    query_idx: int,
    df: pd.DataFrame,
    tfidf_matrix,
    top_k: int = EVAL_TOP_K,
) -> float:
    query_vector = tfidf_matrix[query_idx]
    query_cats = parse_categories(df.iloc[query_idx]["categories"])

    scores = cosine_similarity(query_vector, tfidf_matrix).flatten() # explicitly compute cosine similarity between query and all papers
    top_indices = np.argsort(scores)[::-1]
    retrieved = [idx for idx in top_indices if idx != query_idx][:top_k]

    matches = sum(
        1
        for idx in retrieved
        if parse_categories(df.iloc[idx]["categories"]) & query_cats
    )
    return matches / top_k # same as embedding precision@k but using tf-idf cosine similarity instead of FAISS



def evaluate_embeddings(
    df: pd.DataFrame,
    index,
    tokenizer,
    model,
    device: torch.device,
    sample_size: int = SAMPLE_SIZE,
) -> float:
    rng = np.random.default_rng(42)
    sample_size = min(sample_size, len(df))
    sample_indices = rng.choice(len(df), size=sample_size, replace=False)

    scores = []
    for idx in tqdm(sample_indices, desc="Embedding Precision@5"):
        scores.append(
            precision_at_k(
                query_idx=int(idx),
                df=df,
                index=index,
                tokenizer=tokenizer,
                model=model,
                device=device,
            )
        )

    return float(np.mean(scores))


def evaluate_tfidf(df: pd.DataFrame, sample_size: int = SAMPLE_SIZE) -> float:
    _, tfidf_matrix = load_tfidf_matrix()

    rng = np.random.default_rng(42)
    sample_size = min(sample_size, len(df))
    sample_indices = rng.choice(len(df), size=sample_size, replace=False)

    scores = []
    for idx in tqdm(sample_indices, desc="TF-IDF Precision@5"):
        scores.append(
            precision_at_k_tfidf(
                query_idx=int(idx),
                df=df,
                tfidf_matrix=tfidf_matrix,
            )
        )

    return float(np.mean(scores))



def main():
    device = get_device()
    print(f"Using device: {device}")

    df = load_df()
    index = load_index()
    tokenizer, model = load_model(device)

    print(f"Loaded {len(df):,} papers")
    print(f"FAISS index size: {index.ntotal:,}")
    print("\nSample recommendations:\n")

    results = recommend(
        text_input=df["abstract"].iloc[0], # using the first paper's abstract as a sample query
        df=df,
        index=index,
        tokenizer=tokenizer,
        model=model,
        device=device,
        top_k=TOP_K,
    )

    if results.empty:
        print("No recommendations found.")
    else:
        print(results.to_string(index=False))

    print("\nRunning embedding evaluation...")
    mean_precision = evaluate_embeddings(
        df=df,
        index=index,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )
    print(f"Embedding Mean Precision@{EVAL_TOP_K}: {mean_precision:.4f}")

    if RUN_TFIDF_BASELINE:
        print("\nRunning TF-IDF evaluation...")
        mean_tfidf_precision = evaluate_tfidf(df)
        print(f"TF-IDF Mean Precision@{EVAL_TOP_K}:    {mean_tfidf_precision:.4f}")
        print(f"Embedding Mean Precision@{EVAL_TOP_K}: {mean_precision:.4f}")
        print(f"Improvement:                            +{mean_precision - mean_tfidf_precision:.4f}")

if __name__ == "__main__":
    main()
