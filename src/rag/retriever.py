from pathlib import Path
import sqlite3
import pandas as pd

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.database import DATABASE_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHROMA_PATH = PROJECT_ROOT / "data" / "chroma_db"

# ========================== Helper funtions ==========================

def get_yearly_totals():
    conn = sqlite3.connect(DATABASE_PATH)
    df = pd.read_sql(
        "SELECT year, COUNT(*) AS total FROM papers GROUP BY year",
        conn
    )
    conn.close()
    return df

def get_topic_labels():
    conn = sqlite3.connect(DATABASE_PATH)
    df = pd.read_sql(
        "SELECT topic_id, topic_name, keywords FROM topics",
        conn
    )
    conn.close()
    return df


# ======================== Adaptive retrieval function ==========================

def retrieve_papers_adaptive(query, relative_margin=1.35, min_results=5, max_k=100):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=str(CHROMA_PATH), embedding_function=embedding_model)

    results = vector_db.similarity_search_with_score(query, k=max_k)
    if not results:
        return pd.DataFrame()

    best_distance = results[0][1]
    cutoff = best_distance * relative_margin

    filtered = [(doc, d) for doc, d in results if d <= cutoff]

    if len(filtered) < min_results:
        filtered = results[:min_results]

    distance_map = {doc.metadata["source_file"]: d for doc, d in filtered}
    source_files = list(distance_map.keys())

    conn = sqlite3.connect(DATABASE_PATH)
    placeholders = ",".join(["?"] * len(source_files))
    df = pd.read_sql(
        f"""
        SELECT source_file, title, authors, abstract, keywords, year, topic_id
        FROM papers
        WHERE source_file IN ({placeholders})
        """,
        conn,
        params=source_files
    )
    conn.close()

    df["distance"] = df["source_file"].map(distance_map)
    return df.sort_values("distance").reset_index(drop=True)

