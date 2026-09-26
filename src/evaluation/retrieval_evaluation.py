# Evaluation script for the retrieval module when it uses a threshold. This script evaluates the retrieval performance of the system using precision, recall, and accuracy metrics.

import sqlite3
import pandas as pd

from src.database import DATABASE_PATH
from src.retriever import retrieve_papers_by_threshold


def evaluate_retrieval(query, expected_topic_id, max_distance=1.4):
    retrieved_df = retrieve_papers_by_threshold(query, max_distance=max_distance)

    conn = sqlite3.connect(DATABASE_PATH)
    all_papers = pd.read_sql("SELECT source_file, topic_id FROM papers", conn)
    conn.close()

    retrieved_ids = set(retrieved_df["source_file"])
    relevant_ids = set(all_papers[all_papers["topic_id"] == expected_topic_id]["source_file"])
    all_ids = set(all_papers["source_file"])

    tp = len(retrieved_ids & relevant_ids)
    fp = len(retrieved_ids - relevant_ids)
    fn = len(relevant_ids - retrieved_ids)
    tn = len(all_ids - retrieved_ids - relevant_ids)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy = (tp + tn) / len(all_ids) if len(all_ids) > 0 else 0

    return {
        "query": query,
        "expected_topic_id": expected_topic_id,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "accuracy": round(accuracy, 3),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "retrieved_df": retrieved_df,  # kept for manual spot-check below
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
        result = evaluate_retrieval(query, expected_topic)
        results.append(result)

        print(f"\n=== {query} ===")
        print(f"Precision: {result['precision']}  Recall: {result['recall']}  Accuracy: {result['accuracy']}")
        print(f"TP={result['tp']} FP={result['fp']} FN={result['fn']} TN={result['tn']}")

    avg_precision = sum(r["precision"] for r in results) / len(results)
    avg_recall = sum(r["recall"] for r in results) / len(results)
    avg_accuracy = sum(r["accuracy"] for r in results) / len(results)

    print(f"\n=== Aggregate ===")
    print(f"Avg Precision: {avg_precision:.3f}  Avg Recall: {avg_recall:.3f}  Avg Accuracy: {avg_accuracy:.3f}")