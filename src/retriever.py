from pathlib import Path
import sqlite3
from turtle import distance
import pandas as pd

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.database import DATABASE_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[1]

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

# ========================== Main function ==========================

def retrieve_papers(query, k=10):


    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


    vector_db = Chroma(
        persist_directory=str(CHROMA_PATH),
        embedding_function=embedding_model
    )


    results = vector_db.similarity_search(
        query,
        k=k
    )


    source_files = [
        doc.metadata["source_file"]
        for doc in results
    ]


    if not source_files:
        return []


    conn = sqlite3.connect(
        DATABASE_PATH
    )


    placeholders = ",".join(
        ["?"] * len(source_files)
    )


    query_sql = f"""
    SELECT
        source_file,
        title,
        authors,
        abstract,
        keywords,
        year,
        topic_id

    FROM papers

    WHERE source_file IN ({placeholders})
    """


    df = pd.read_sql(
        query_sql,
        conn,
        params=source_files
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

# =============== Temporary function for testing purposes ===============

def retrieve_papers_by_threshold(query, max_distance=1.4, max_k=100):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=str(CHROMA_PATH), embedding_function=embedding_model)

    results = vector_db.similarity_search_with_score(query, k=max_k)
    filtered = [(doc, distance) for doc, distance in results if distance <= max_distance]

    if not filtered:
        return pd.DataFrame()

    distance_map = {doc.metadata["source_file"]: distance for doc, distance in filtered}
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

if __name__ == "__main__":
    from src.retriever import retrieve_papers_by_threshold

    test_queries = [
        "supervised api intelligent expert smart learning",
        "cnn recognition classification classify convolutional neural",
        "prediction predict regression classify svm rmse",
    ]

    for query in test_queries:
        print(f"\n=== Query: {query} ===")
        df = retrieve_papers_by_threshold(query, max_distance=1.4)
        
        if df.empty:
            print("No papers matched.")
            continue

        print(f"{len(df)} paper(s) matched\n")

        for _, row in df.iterrows():
            print(
                f"dist={row['distance']:.3f}  "
                f"year={row['year']}  "
                f"topic_id={row['topic_id']}  "
                f"{row['title']}"
            )
