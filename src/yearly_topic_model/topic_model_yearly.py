from datetime import datetime

import pandas as pd
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic

from src.database import (
    get_available_years,
    load_papers_by_year,
    save_year_topics,
    load_year_topics,
)
from src.knowledge_base.topic_model import (
    create_umap_model,
    create_hdbscan_model,
    create_vectorizer,
    create_ctfidf_model,
    create_representation_model,
    get_embeddings_for_papers,
)


def run_topic_model_for_year(year):
    papers = load_papers_by_year(year)

    if papers.empty:
        raise ValueError(f"No papers found for year {year}.")

    documents = papers["document"].tolist()
    source_files = papers["source_file"].tolist()

    embeddings = get_embeddings_for_papers(source_files)
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=create_umap_model(),
        hdbscan_model=create_hdbscan_model(),
        vectorizer_model=create_vectorizer(),
        ctfidf_model=create_ctfidf_model(),
        representation_model=create_representation_model(),
        verbose=True,
    )

    topics, _ = topic_model.fit_transform(documents, embeddings=embeddings)

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

    topic_info = topic_model.get_topic_info()
    created_at = datetime.now().isoformat()
    rows = []

    for _, row in topic_info.iterrows():
        if row["Topic"] == -1:
            continue

        words = topic_model.get_topic(row["Topic"])
        keywords = ", ".join(word for word, _ in words)
        top_words = [word for word, _ in words[:3]]
        topic_name = " / ".join(w.capitalize() for w in top_words)

        rows.append({
            "year": year,
            "topic_id": int(row["Topic"]),
            "topic_name": topic_name,
            "keywords": keywords,
            "paper_count": int(row["Count"]),
            "created_at": created_at,
        })

    topics_df = pd.DataFrame(rows)

    save_year_topics(year, topics_df)

    return load_year_topics(year)

# For evaluation purposes, we can also have a function 
# that fits the topic model for a given year and returns 
# the model object and related data without saving to 
# the database. This can be useful for testing or further analysis.
def fit_topic_model_for_year(year):
    """Fits and returns the BERTopic model object + documents, without saving to DB."""
    papers = load_papers_by_year(year)

    if papers.empty:
        raise ValueError(f"No papers found for year {year}.")

    documents = papers["document"].tolist()
    source_files = papers["source_file"].tolist()

    embeddings = get_embeddings_for_papers(source_files)
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=create_umap_model(),
        hdbscan_model=create_hdbscan_model(),
        vectorizer_model=create_vectorizer(),
        ctfidf_model=create_ctfidf_model(),
        representation_model=create_representation_model(),
        verbose=True,
    )

    topics, _ = topic_model.fit_transform(documents, embeddings=embeddings)

    if -1 in topics:
        topics = topic_model.reduce_outliers(
            documents, topics, strategy="embeddings", threshold=0.1,
        )
        topic_model.update_topics(
            documents, topics=topics,
            vectorizer_model=topic_model.vectorizer_model,
            ctfidf_model=topic_model.ctfidf_model,
            representation_model=topic_model.representation_model,
        )

    return topic_model, papers, documents, topics