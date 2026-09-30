from pathlib import Path
import json
import pandas as pd
import streamlit as st

def preprocess_metadata(metadata_folder: Path) -> tuple[pd.DataFrame, dict]:
    processed_records = []
    skipped_records = []
    already_done_count = 0

    json_files = sorted(metadata_folder.glob("*.json"))

    for json_file in json_files:

        with open(json_file, "r", encoding="utf-8") as f:
            paper = json.load(f)

        if paper.get("preprocessed", False):
            already_done_count += 1  # not part of this run — exclude from totals
            continue

        if paper.get("status") != "success":
            skipped_records.append({"file": json_file.name, "reason": f"status={paper.get('status')}"})
            continue

        title = paper.get("title")
        authors = paper.get("authors", [])
        abstract = paper.get("abstract")
        year = paper.get("year")
        keywords = paper.get("keywords", [])
        source_file = paper.get("source_file")

        if not title or not abstract or not year:
            skipped_records.append({"file": json_file.name, "reason": "missing title/abstract/year"})
            continue

        year = str(year)

        import re

        match = re.search(r"\d{4}", year)

        if not match:
            skipped_records.append({"file": json_file.name, "reason": f"invalid year={year}"})
            continue

        year = int(match.group())

        document = title.strip() + "\n\nAbstract:\n" + abstract.strip()

        processed_records.append({

            "title": title.strip(),
            "authors": authors,
            "abstract": abstract.strip(),
            "keywords": keywords,
            "year": year,
            "document": document,
            "source_file": source_file

        })

        paper["preprocessed"] = True

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(paper, f, indent=2, ensure_ascii=False)

    meta_df = pd.DataFrame(processed_records)

    if not meta_df.empty:
        meta_df.drop_duplicates(subset=["title"], inplace=True)

    summary = {
        "total": len(processed_records) + len(skipped_records),  # this run only
        "success": len(meta_df),
        "failure": len(skipped_records),
        "skipped_details": skipped_records,
    }

    return meta_df, summary