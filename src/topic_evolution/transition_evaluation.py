import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.database import load_papers_by_year
from src.yearly_topic_model.topic_model_yearly import fit_topic_model_for_year
from src.topic_model import get_embeddings_for_papers, create_vectorizer, create_ctfidf_model


GROWTH_THRESHOLD = 0.2  # 20% relative change — tune as needed

# ========= Topic transformation by similarity ========= 

def transform_by_similarity(topic_model_from, embeddings_to, similarity_threshold=0.3):
    """
    Assigns each new document to the most similar year_from topic based on
    cosine similarity to topic embeddings, instead of HDBSCAN's approximate_predict
    (which is overly strict for small training sets).
    """
    topic_embeddings = topic_model_from.topic_embeddings_  # shape: (n_topics_incl_outlier, dim)
    topic_ids = sorted(topic_model_from.get_topics().keys())  # includes -1 if present

    valid_mask = [tid != -1 for tid in topic_ids]
    valid_topic_ids = [tid for tid, v in zip(topic_ids, valid_mask) if v]
    valid_embeddings = topic_embeddings[[i for i, v in enumerate(valid_mask) if v]]

    sims = cosine_similarity(embeddings_to, valid_embeddings)  # (n_docs, n_topics)

    assigned = []
    for row in sims:
        best_idx = row.argmax()
        best_sim = row[best_idx]
        if best_sim >= similarity_threshold:
            assigned.append(valid_topic_ids[best_idx])
        else:
            assigned.append(-1)

    return assigned, sims

def _classify_change(share_from, share_to):
    if share_from == 0 and share_to > 0:
        return "new"  # shouldn't normally hit this path — see outlier handling below
    if share_to == 0:
        return "disappeared"

    relative_change = (share_to - share_from) / share_from if share_from > 0 else float("inf")

    if relative_change > GROWTH_THRESHOLD:
        return "grew"
    elif relative_change < -GROWTH_THRESHOLD:
        return "declined"
    else:
        return "persisted"

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity



# ========= Evaluation of topic evolution between two years =========

def evaluate_transition(year_from, year_to):
    """
    Fits BERTopic on year_from, then transforms year_to's papers through
    that fixed model to see how year_from's topics fared in year_to.
    """
    topic_model_from, papers_from, docs_from, topics_from = fit_topic_model_for_year(year_from)

    papers_to = load_papers_by_year(year_to)
    if papers_to.empty:
        raise ValueError(f"No papers found for year {year_to}.")

    docs_to = papers_to["document"].tolist()
    source_files_to = papers_to["source_file"].tolist()
    embeddings_to = get_embeddings_for_papers(source_files_to)

    new_topics, sims = transform_by_similarity(topic_model_from, embeddings_to, similarity_threshold=0.3)

    total_from = len(papers_from)
    total_to = len(papers_to)

    topic_info_from = topic_model_from.get_topic_info()

    rows = []
    for _, row in topic_info_from.iterrows():
        topic_id = row["Topic"]
        if topic_id == -1:
            continue

        count_from = row["Count"]
        count_to = sum(1 for t in new_topics if t == topic_id)

        share_from = count_from / total_from
        share_to = count_to / total_to

        words = topic_model_from.get_topic(topic_id)
        keywords = ", ".join(word for word, _ in words[:6])

        rows.append({
            "topic_id": int(topic_id),
            "keywords": keywords,
            f"count_{year_from}": count_from,
            f"count_{year_to}": count_to,
            f"share_{year_from}": round(share_from * 100, 1),
            f"share_{year_to}": round(share_to * 100, 1),
            "status": _classify_change(share_from, share_to),
        })

    evolution_df = pd.DataFrame(rows).sort_values(f"count_{year_to}", ascending=False)

    # -----------------------------------------
    # Papers in year_to that fit NONE of year_from's topics
    # -----------------------------------------
    outlier_mask = [t == -1 for t in new_topics]
    outlier_indices = [i for i, is_outlier in enumerate(outlier_mask) if is_outlier]
    outlier_papers = papers_to.iloc[outlier_indices].reset_index(drop=True)

    emerging_keywords = None
    if len(outlier_papers) >= 4:  # matches your min_cluster_size floor
        outlier_docs = outlier_papers["document"].tolist()
        vectorizer = create_vectorizer()
        ctfidf = create_ctfidf_model()

        try:
            X = vectorizer.fit_transform(outlier_docs)
            X_c = ctfidf.fit_transform(X)
            scores = X_c.toarray().sum(axis=0)
            terms = vectorizer.get_feature_names_out()
            top_indices = scores.argsort()[::-1][:10]
            emerging_keywords = ", ".join(terms[i] for i in top_indices)
        except Exception:
            emerging_keywords = None  # too few distinct terms, leave for manual review

    return {
        "evolution_df": evolution_df,
        "outlier_papers": outlier_papers,   # papers not fitting any year_from topic
        "emerging_keywords": emerging_keywords,  # candidate name for the emerging cluster, if computable
    }