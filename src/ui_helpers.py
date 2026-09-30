import streamlit as st
import pandas as pd

def show_summary_and_preview(df, title, total=None, success=None, failure=None, preview_rows=5):
    st.subheader(title)

    if total is not None:
        st.write(f"**Total:** {total}  |  **Success:** {success}  |  **Failure:** {failure}")

    if df.empty:
        st.info("No records to display.")
        return

    st.dataframe(df.head(preview_rows), width="stretch")

    if len(df) > preview_rows:
        with st.expander(f"Show all {len(df)} record(s)"):
            st.dataframe(df, width="stretch")

def build_success_df(results: list) -> pd.DataFrame:
    """Extracts successfully processed papers from this run's results for preview."""
    success_records = [r for r in results if r["status"] == "success"]
    return pd.DataFrame([
        {
            "source_file": r.get("source_file"),
            "title": r.get("title"),
            "authors": r.get("authors"),
            "year": r.get("year"),
        }
        for r in success_records
    ])