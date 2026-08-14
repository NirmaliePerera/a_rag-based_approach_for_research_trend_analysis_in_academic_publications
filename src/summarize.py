import os
import hashlib
import sqlite3
from datetime import datetime

from google import genai

from src.database import DATABASE_PATH

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

MODEL_NAME = "gemini-3.5-flash"  # check ai.google.dev for current model names before deploying


def _ensure_cache_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS summary_cache (
            query TEXT NOT NULL,
            paper_ids_hash TEXT NOT NULL,
            summary TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (query, paper_ids_hash)
        )
    """)
    conn.commit()


def _hash_paper_ids(papers_df):
    ids = sorted(papers_df["source_file"].tolist())
    return hashlib.sha256(",".join(ids).encode()).hexdigest()


def _get_cached_summary(conn, query, paper_ids_hash):
    row = conn.execute(
        "SELECT summary FROM summary_cache WHERE query = ? AND paper_ids_hash = ?",
        (query, paper_ids_hash),
    ).fetchone()
    return row[0] if row else None


def _save_summary(conn, query, paper_ids_hash, summary):
    conn.execute(
        """
        INSERT INTO summary_cache VALUES (?, ?, ?, ?)
        ON CONFLICT(query, paper_ids_hash) DO UPDATE SET
            summary = excluded.summary,
            created_at = excluded.created_at
        """,
        (query, paper_ids_hash, summary, datetime.now().isoformat()),
    )
    conn.commit()


def build_context(papers_df, max_papers=10, max_abstract_chars=500):
    entries = []

    for _, row in papers_df.head(max_papers).iterrows():
        abstract = str(row["abstract"])[:max_abstract_chars]

        entries.append(
            f"Year: {row['year']}\n"
            f"Abstract: {abstract}"
        )

    return "\n\n---\n\n".join(entries)


def generate_summary(papers_df, query):
    if papers_df.empty:
        return "No related papers found to summarize."

    conn = sqlite3.connect(DATABASE_PATH)
    _ensure_cache_table(conn)

    paper_ids_hash = _hash_paper_ids(papers_df)

    cached = _get_cached_summary(conn, query, paper_ids_hash)
    if cached:
        conn.close()
        return cached

    context = build_context(papers_df)

    prompt = f"""You are analyzing research trends for the topic: "{query}"

Based on the following paper abstracts (with publication years), write a concise trend summary covering:
1. What this research area is about, in plain terms.
2. How the focus or approach appears to have evolved across the years represented.
3. Any notable shift, emerging sub-theme, or turning point visible in the papers.

Keep it to 3-5 sentences. Do not list papers individually — synthesize across them.

Abstracts:
{context}
"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )
        summary = response.text
        _save_summary(conn, query, paper_ids_hash, summary)

    except Exception as e:
        summary = f"Summary generation failed: {e}"

    conn.close()
    return summary