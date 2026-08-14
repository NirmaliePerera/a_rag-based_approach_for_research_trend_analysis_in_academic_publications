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


# =============== Temporary function for testing purposes ===============

def retrieve_papers_by_threshold(query, max_distance=1.1, max_k=100):
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = Chroma(
        persist_directory=str(CHROMA_PATH),
        embedding_function=embedding_model
    )

    results = vector_db.similarity_search_with_score(query, k=max_k)
    filtered = [doc for doc, distance in results if distance <= max_distance]

    source_files = [doc.metadata["source_file"] for doc in filtered]

    if not source_files:
        return pd.DataFrame()

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

    return df

if __name__ == "__main__":
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=str(CHROMA_PATH), embedding_function=embedding_model)

    for query in [
        "supervised api intelligent expert smart learning",
        "cnn recognition classification classify convolutional neural",
        "prediction predict regression classify svm rmse",
    ]:
        print(f"\n--- {query} ---")
        results = vector_db.similarity_search_with_score(query, k=15)
        for doc, distance in results:
            print(round(distance, 3), doc.metadata["year"], doc.metadata["title"])

