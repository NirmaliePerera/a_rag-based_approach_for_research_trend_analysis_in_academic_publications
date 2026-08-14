from pathlib import Path
import json
import pandas as pd
import streamlit as st

def preprocess_metadata(metadata_folder: Path) -> pd.DataFrame:
    """
    Preprocess newly extracted metadata only.
    Marks processed JSON files with:
        "preprocessed": true
    """

    processed_records = []

    json_files = sorted(metadata_folder.glob("*.json"))

    for json_file in json_files:

        with open(json_file, "r", encoding="utf-8") as f:
            paper = json.load(f)

        # st.write({
        #     "file": json_file.name,
        #     "status": paper.get("status"),
        #     "preprocessed": paper.get("preprocessed"),
        #     "title": bool(paper.get("title")),
        #     "abstract": bool(paper.get("abstract")),
        #     "year": paper.get("year")
        # })

        # ---------------------------------------
        # Skip failed extractions
        # ---------------------------------------

        if paper.get("status") != "success":
            print(f"{json_file.name}: skipped (status={paper.get('status')})")
            continue

        # ---------------------------------------
        # Skip already preprocessed papers
        # ---------------------------------------

        if paper.get("preprocessed", False):
            print(f"{json_file.name}: skipped (already preprocessed)")
            continue

        # ---------------------------------------
        # Required fields
        # ---------------------------------------

        title = paper.get("title")
        authors = paper.get("authors", [])
        abstract = paper.get("abstract")
        year = paper.get("year")
        keywords = paper.get("keywords", [])
        source_file = paper.get("source_file")

        # ---------------------------------------
        # Ignore incomplete metadata
        # ---------------------------------------

        if not title or not abstract or not year:
            print(f"{json_file.name}: skipped (missing title/abstract/year)")
            continue

        # ---------------------------------------
        # Normalize year
        # ---------------------------------------

        year = str(year)

        import re

        match = re.search(r"\d{4}", year)

        if not match:
            print(f"{json_file.name}: skipped (invalid year={year})")
            continue

        year = int(match.group())

        # ---------------------------------------
        # Create document
        # ---------------------------------------

        document = (
            title.strip()
            + "\n\nAbstract:\n"
            + abstract.strip()
        )

        processed_records.append({

            "title": title.strip(),
            "authors": authors,
            "abstract": abstract.strip(),
            "keywords": keywords,
            "year": year,
            "document": document,
            "source_file": source_file

        })

        # ---------------------------------------
        # Mark JSON as preprocessed
        # ---------------------------------------

        paper["preprocessed"] = True

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(
                paper,
                f,
                indent=2,
                ensure_ascii=False
            )


    meta_df = pd.DataFrame(processed_records)

    
    # ---------------------------------------
    # Remove duplicate papers
    # ---------------------------------------

    if not meta_df.empty:

        meta_df.drop_duplicates(
            subset=["title"],
            inplace=True
        )

    return meta_df