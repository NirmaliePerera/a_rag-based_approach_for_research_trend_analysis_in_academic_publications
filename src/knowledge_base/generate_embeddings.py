from pathlib import Path
import sqlite3
import pandas as pd

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from src.database import DATABASE_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHROMA_PATH = PROJECT_ROOT / "data" / "chroma_db"


def generate_embeddings():

    conn = sqlite3.connect(DATABASE_PATH)

    cursor = conn.cursor()

    cursor.execute("""
        SELECT source_file, title, authors, year, document
        FROM papers
        WHERE embedded = 0
    """)

    rows = cursor.fetchall()

    if not rows:

        conn.close()
        return pd.DataFrame()  # empty — nothing to embed this run

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = Chroma(
        persist_directory=str(CHROMA_PATH),
        embedding_function=embedding_model
    )

    documents = []
    ids = []
    embedded_records = []

    for source_file, title, authors, year, document in rows:

        documents.append(

            Document(

                page_content=document,

                metadata={
                    "source_file": source_file,
                    "title": title,
                    "authors": authors,
                    "year": year
                }

            )

        )

        ids.append(source_file)
        embedded_records.append({
            "source_file": source_file,
            "title": title,
            "authors": authors,
            "year": year
        })

    vector_db.add_documents(
        documents=documents,
        ids=ids
    )

    cursor.execute("""
        UPDATE papers
        SET embedded = 1
        WHERE embedded = 0
    """)

    conn.commit()

    conn.close()

    return pd.DataFrame(embedded_records)