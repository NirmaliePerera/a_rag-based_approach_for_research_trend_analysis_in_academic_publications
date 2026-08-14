from pathlib import Path
import sqlite3
from datetime import datetime
from turtle import st

import numpy as np
import pandas as pd
import spacy

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.database import DATABASE_PATH

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_PATH = PROJECT_ROOT / "data" / "chroma_db"


# -------------------------------------------------------------
# Helper functions for topic modeling
# -------------------------------------------------------------

def create_umap_model():
    return UMAP(
        n_neighbors=10,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
        n_jobs=1,  # forces determinism across machines/core counts
    )


def create_hdbscan_model():
    return HDBSCAN(
        min_cluster_size=4,
        min_samples=2,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )


def create_vectorizer():
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    domain_stop_words = {
        "learning", "dataset", "datasets", "model", "models", "classification",
        "data", "paper", "proposed", "using", "based", "results", "ai", "approach",
        "method", "methods", "system", "analysis", "performance", "trained",
        "training", "article", "study", "present", "presents", "classifier", "classifiers",
    }
    for word in domain_stop_words:
        nlp.vocab[word].is_stop = True

    def lemmatize_and_clean(text):
        doc = nlp(text)
        return [
            token.lemma_.lower()
            for token in doc
            if token.is_alpha and not token.is_stop
        ]

    return CountVectorizer(
        analyzer=lemmatize_and_clean,
        ngram_range=(1, 2),
        min_df=2,
    )


def create_representation_model():
    return [
        KeyBERTInspired(),
        MaximalMarginalRelevance(diversity=0.6),
    ]


def create_ctfidf_model():
    return ClassTfidfTransformer(reduce_frequent_words=True)


# -------------------------------------------------------------
# Fetch embeddings from Chroma, order-safe
# -------------------------------------------------------------

def get_embeddings_for_papers(source_files: list[str]) -> np.ndarray:
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_db = Chroma(
        persist_directory=str(CHROMA_PATH),
        embedding_function=embedding_model,
    )

    results = vector_db.get(ids=source_files, include=["embeddings"])

    id_to_embedding = dict(zip(results["ids"], results["embeddings"]))

    missing = [sf for sf in source_files if sf not in id_to_embedding]
    if missing:
        raise ValueError(f"{len(missing)} papers missing embeddings in Chroma: {missing[:5]}...")

    # Re-assemble in the SAME order as source_files, not Chroma's return order
    return np.array([id_to_embedding[sf] for sf in source_files])


# --------------------------------------------------------
# Main function for topic modeling — modeling all papers
# --------------------------------------------------------

def run_topic_model():
    conn = sqlite3.connect(DATABASE_PATH)

    papers = pd.read_sql(
        """
        SELECT source_file, title, document
        FROM papers
        ORDER BY source_file
        """,
        conn,
    )

    documents = papers["document"].tolist()
    source_files = papers["source_file"].tolist()

    # -----------------------------
    # Embeddings — pulled from Chroma, order-matched to `documents`
    # -----------------------------
    embeddings = get_embeddings_for_papers(source_files)

    # Still needed for reduce_outliers(strategy="embeddings") and KeyBERTInspired
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # -----------------------------
    # BERTopic
    # -----------------------------
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=create_umap_model(),
        hdbscan_model=create_hdbscan_model(),
        vectorizer_model=create_vectorizer(),
        ctfidf_model=create_ctfidf_model(),
        representation_model=create_representation_model(),
        verbose=True,
    )

    # -----------------------------
    # Temporary: Streamlit debug output for analyzer and embeddings shape
    # -----------------------------
    # import streamlit as st
    # st.write("Analyzer:", topic_model.vectorizer_model.analyzer)
    # st.write("Embeddings shape:", embeddings.shape, "vs documents:", len(documents))

    # -----------------------------
    # Temporary: Streamlit debug output for analyzer and embeddings shape
    # -----------------------------


    topics, _ = topic_model.fit_transform(documents, embeddings=embeddings)

    # TEMP DEBUG — check keywords immediately after initial fit, before outlier reduction
    # st.write("Topic 0 keywords BEFORE reduce_outliers:", topic_model.get_topic(0))
    # st.write("Topic info BEFORE reduce_outliers:", topic_model.get_topic_info())

    # -----------------------------
    # Reduce outliers (only if there are any)
    # -----------------------------
    if -1 in topics:
        topics = topic_model.reduce_outliers(
            documents,
            topics,
            strategy="embeddings",
            threshold=0.1,
        )

        topic_model.update_topics(
            documents,
            topics=topics,
            vectorizer_model=topic_model.vectorizer_model,
            ctfidf_model=topic_model.ctfidf_model,
            representation_model=topic_model.representation_model,
        )
    
    # TEMP DEBUG — check keywords after update_topics
    # st.write("Topic 0 keywords AFTER update_topics:", topic_model.get_topic(0))

    # -----------------------------
    # Save to SQLite
    # -----------------------------
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS topics(
            topic_id INTEGER PRIMARY KEY,
            topic_name TEXT,
            keywords TEXT,
            created_at TEXT
        )
    """)

    cursor.execute("DELETE FROM topics")
    cursor.execute("UPDATE papers SET topic_id = NULL")

    topic_info = topic_model.get_topic_info()
    created_at = datetime.now().isoformat()

    for _, row in topic_info.iterrows():
        if row["Topic"] == -1:
            continue

        words = topic_model.get_topic(row["Topic"])
        keywords = ", ".join(word for word, _ in words)

# ------ Newly added code to build a readable topic name from the top 3 keywords instead of just "Topic N" ------
        top_words = [word for word, _ in words[:3]]
        topic_name = " / ".join(w.capitalize() for w in top_words)

# ------ Modified code to insert topic_name instead of "Topic N" ------
        cursor.execute(
            "INSERT INTO topics VALUES (?, ?, ?, ?)",
            (int(row["Topic"]), topic_name, keywords, created_at),
        )


    for index, topic_id in enumerate(topics):
        cursor.execute(
            "UPDATE papers SET topic_id = ? WHERE source_file = ?",
            (int(topic_id), papers.iloc[index]["source_file"]),
        )

    conn.commit()
    conn.close()

    # TEMP DEBUG — verify what was actually just written, same process, same DB handle
    verify_conn = sqlite3.connect(DATABASE_PATH)
    verify_df = pd.read_sql(
        "SELECT topic_id, topic_name, keywords FROM topics ORDER BY topic_id",
        verify_conn
    )

    verify_conn.close()
    import streamlit as st
    # st.write("Topics written to DB (verified read-back):", verify_df)

    # st.write("DATABASE_PATH resolved to:", DATABASE_PATH, "| exists:", Path(DATABASE_PATH).exists())

    st.divider()
    # st.subheader("🔍 DEBUG: Topics written to DB")
    st.subheader("Generated Topics")
    st.dataframe(verify_df)
    #st.write("DATABASE_PATH:", str(DATABASE_PATH), "| exists:", Path(DATABASE_PATH).exists())
    st.divider()

    return len(topic_info[topic_info["Topic"] != -1])