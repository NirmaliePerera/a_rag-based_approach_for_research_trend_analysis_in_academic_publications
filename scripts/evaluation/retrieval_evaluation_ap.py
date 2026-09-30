# Evaluation script for the retrieval module when it does not use a threshold. 
# This script evaluates the retrieval performance of the system using Average Precision (AP), ROC-AUC, and Precision/Recall at K metrics.
# Use this script to evaluate the retrieval performance of the system without applying a distance threshold. 
# This can be used for the current adaptive retrieval approach.
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
import matplotlib.pyplot as plt # Save PR curve plot

from src.database import DATABASE_PATH
from src.rag.retriever import CHROMA_PATH
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import sqlite3

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "data" / "evaluation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def precision_recall_at_k(y_true, y_score, k):
    order = np.argsort(y_score)[::-1][:k]
    top_k_true = np.array(y_true)[order]
    tp = top_k_true.sum()
    precision_k = tp / k
    recall_k = tp / sum(y_true) if sum(y_true) > 0 else 0
    return precision_k, recall_k


def evaluate_retrieval_no_threshold(query, expected_topic_id):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=str(CHROMA_PATH), embedding_function=embedding_model)

    conn = sqlite3.connect(DATABASE_PATH)
    total_papers = int(pd.read_sql("SELECT COUNT(*) as n FROM papers", conn)["n"][0])
    topic_lookup = pd.read_sql("SELECT source_file, topic_id FROM papers", conn).set_index("source_file")["topic_id"].to_dict()
    conn.close()

    results = vector_db.similarity_search_with_score(query, k=total_papers)

    y_true, y_score = [], []
    for doc, distance in results:
        source_file = doc.metadata["source_file"]
        is_relevant = 1 if topic_lookup.get(source_file) == expected_topic_id else 0
        y_true.append(is_relevant)
        y_score.append(-distance)

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    p_at_5, r_at_5 = precision_recall_at_k(y_true, y_score, 5)
    p_at_10, r_at_10 = precision_recall_at_k(y_true, y_score, 10)

    return {
        "query": query,
        "expected_topic_id": expected_topic_id,
        "average_precision": round(ap, 3),
        "roc_auc": round(auc, 3),
        "precision_at_5": round(p_at_5, 3),
        "recall_at_5": round(r_at_5, 3),
        "precision_at_10": round(p_at_10, 3),
        "recall_at_10": round(r_at_10, 3),
        "precision_curve": precision,
        "recall_curve": recall,
    }


if __name__ == "__main__":
    test_cases = [
        ("cnn recognition convolutional neural classification deep feature detection learning segmentation", 0),
        ("prediction svm predict lstm regression predictive optimize learning machine boost", 1),
        ("ai intelligent personalize service agent nlp text create language feature", 2),
        ("iot classifier nearest wireless location sensor supervised accuracy neighbor range", 3),
    ]

    results = []
    for query, expected_topic in test_cases:
        result = evaluate_retrieval_no_threshold(query, expected_topic)
        results.append(result)
        print(
            f"Topic {expected_topic}: AP={result['average_precision']}, "
            f"AUC={result['roc_auc']}, P@5={result['precision_at_5']}, "
            f"R@5={result['recall_at_5']}, P@10={result['precision_at_10']}, "
            f"R@10={result['recall_at_10']}"
        )

    mean_ap = sum(r["average_precision"] for r in results) / len(results)
    mean_auc = sum(r["roc_auc"] for r in results) / len(results)
    print(f"\nMean Average Precision (mAP): {round(mean_ap, 3)}")
    print(f"Mean ROC-AUC: {round(mean_auc, 3)}")

    # Save table to CSV
    summary_df = pd.DataFrame([
        {
            "topic_id": r["expected_topic_id"],
            "average_precision": r["average_precision"],
            "roc_auc": r["roc_auc"],
            "precision_at_5": r["precision_at_5"],
            "recall_at_5": r["recall_at_5"],
            "precision_at_10": r["precision_at_10"],
            "recall_at_10": r["recall_at_10"],
        }
        for r in results
    ])
    summary_df.to_csv(RESULTS_DIR / "retrieval_evaluation_results.csv", index=False)
    print("\nSaved results table to retrieval_evaluation_results.csv")

# Plot Precision-Recall curves for each topic

    for result in results:
        plt.plot(result["recall_curve"], result["precision_curve"], label=f"Topic {result['expected_topic_id']}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves by Topic (threshold-free)")
    plt.legend()
    plt.savefig(RESULTS_DIR / "pr_curve.png")
    print("Saved PR curve plot to pr_curve.png")