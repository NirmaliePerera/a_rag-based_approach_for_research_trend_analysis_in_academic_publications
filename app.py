import streamlit as st
import pandas as pd
from pathlib import Path

from src.extract_metadata import (
    extract_metadata,
    find_missing_metadata
)

from src.database import (
    append_metadata,
    initialize_database,
    get_available_years
)

from src.topic_model import (
    run_topic_model,
    # run_year_topic_model
)

from src.generate_embeddings import generate_embeddings
from src.preprocess_metadata import preprocess_metadata

from src.yearly_topic_model.topic_model_yearly import run_topic_model_for_year
from src.trend_analysing.keyword_bubble import build_keyword_bubble_chart
from src.ui_helpers import show_summary_and_preview, build_success_df
# --------------------------------------------------
# Streamlit App Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Research Trend Analysis",
    layout="wide"
)
# --------------------------------------------------
# Page control
# --------------------------------------------------

if "page" not in st.session_state:
    st.session_state.page = "main"

if st.session_state.page == "main":
    st.title("RAG-based Topic Trend Analysis")

    st.subheader("Knowledge Base Builder")

    st.write("Metadata Extraction, Preprocessing Metadata, Embedding Generation")


    upload_folder = Path("data/uploaded_pdfs")
    upload_folder.mkdir(parents=True, exist_ok=True)

    existing_pdfs = list(upload_folder.glob("*.pdf"))
    has_pdfs = len(existing_pdfs) > 0

    uploaded_files = st.file_uploader(
        "Upload research papers to build the knowledge base.",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        for file in uploaded_files:

            save_path = upload_folder / file.name

            with open(save_path, "wb") as f:
                f.write(file.getbuffer())

        st.success(f"{len(uploaded_files)} paper(s) uploaded successfully.")

    progress = st.progress(0)
    status = st.empty()

    # ---------------------------------------------------------------------
    # Metadata Extraction
    # ---------------------------------------------------------------------

    metadata_folder = Path("data/metadata")
    metadata_folder.mkdir(parents=True, exist_ok=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Extract Metadata", use_container_width=True):
            if not has_pdfs:
                status.error("No PDFs found. Please upload PDF files first.")
            else:
                progress = st.progress(0)
                status = st.empty()

                def update_progress(current, total, filename):
                    progress.progress(current / total)
                    status.text(f"Processing {current}/{total}: {filename}")

                try:
                    summary = extract_metadata(upload_folder, progress_callback=update_progress)
                    progress.progress(1.0)
                    status.empty()

                    success_df = build_success_df(summary["results"])

                    show_summary_and_preview(
                        success_df,
                        title="Metadata Extraction Results",
                        total=summary["success"] + summary["failed"],
                        success=summary["success"],
                        failure=summary["failed"],
                    )

                    if summary["failed"] > 0:
                        with st.expander(f"Show {summary['failed']} failed paper(s)"):
                            for item in summary["failed_files"]:
                                st.error(f"{item['source_file']}\n\n{item['error']}")

                except Exception as e:
                    status.error(str(e))

    # ---------------------------------------------------------------------
    # Check Missing Metadata Button
    # ---------------------------------------------------------------------

    with col2:

        if st.button("Check Missing Metadata", use_container_width=True):

            missing = find_missing_metadata(
                upload_folder,
                metadata_folder
            )

            st.session_state["missing_metadata"] = missing

        # Show result if it has already been checked
        if "missing_metadata" in st.session_state:

            missing = st.session_state["missing_metadata"]

            if not missing:

                st.success("All uploaded PDFs have metadata.")

            else:

                st.warning(f"{len(missing)} paper(s) do not have metadata.")

                for paper in missing:
                    st.write(f"• {paper}.pdf")

                if st.button("Extract Missing Metadata", key="extract_missing_metadata"):
                    progress = st.progress(0)
                    status = st.empty()

                    def update_progress(current, total, filename):
                        progress.progress(current / total)
                        status.text(f"Processing {current}/{total}: {filename}")

                    summary = extract_metadata(upload_folder, progress_callback=update_progress)
                    st.session_state["summary"] = summary

                    progress.progress(1.0)
                    status.empty()

                    success_df = build_success_df(summary["results"])

                    show_summary_and_preview(
                        success_df,
                        title="Missing Metadata Extraction Results",
                        total=summary["success"] + summary["failed"],
                        success=summary["success"],
                        failure=summary["failed"],
                    )

    # ---------------------------------------------------------------------
    # Display extraction summary
    # ---------------------------------------------------------------------

    if "summary" in st.session_state:

        summary = st.session_state["summary"]

        if summary["failed"] == 0:

            status.success(
                f"""
                Metadata extraction completed.

                Successful : {summary['success']}
                Failed : 0
                """
            )

        else:

            status.warning(
                f"""
                Metadata extraction completed with errors.

                Successful : {summary['success']}
                Failed : {summary['failed']}
                """
            )

            with st.expander("View failed papers"):

                for item in summary["failed_files"]:

                    st.error(
                        f"{item['source_file']}\n\n{item['error_type']}"
                    )

                    with st.expander("Technical details"):

                        st.code(item["error"])



    # ---------------------------------------------------------------------
    # Retry failed papers
    # ---------------------------------------------------------------------

    if (
        "summary" in st.session_state
        and st.session_state["summary"]["failed"] > 0
    ):

        if st.button("Retry Failed Papers"):

            progress = st.progress(0)
            status = st.empty()

            def update_progress(current, total, filename):
                progress.progress(current / total)
                status.text(f"Processing {current}/{total}: {filename}")

            try:

                summary = extract_metadata(
                    upload_folder,
                    progress_callback=update_progress
                )

                st.session_state["summary"] = summary

                progress.progress(1.0)

                status.success("Retry completed.")

            except Exception as e:

                status.error(str(e))


    # ---------------------------------------------------------------------
    # Second row for Preprocessing and Embedding
    # ---------------------------------------------------------------------

    col1, col2 = st.columns(2)

    # ---------------------------------------------------------------------
    # Preprocess Metadata Button - before saving in SQLite
    # ---------------------------------------------------------------------

    with col1:

        if st.button(
            "Preprocess Metadata",
            use_container_width=True
        ):

            meta_df, summary = preprocess_metadata(metadata_folder)

            if meta_df.empty:

                st.info("No new papers to preprocess.")

            else:

                append_metadata(meta_df)

                show_summary_and_preview(
                    meta_df,
                    title="Metadata Preprocessing Results",
                    total=summary["total"],
                    success=summary["success"],
                    failure=summary["failure"],
                )

                if summary["failure"] > 0:
                    with st.expander(f"Show {summary['failure']} skipped/failed record(s)"):
                        st.dataframe(pd.DataFrame(summary["skipped_details"]), width="stretch")

    # ---------------------------------------------------------------------
    # Generate Embeddings Button
    # ---------------------------------------------------------------------

    with col2:

        if st.button(
            "Generate Embeddings", 
            use_container_width=True
        ):

            embedded_df = generate_embeddings()

            show_summary_and_preview(
                embedded_df,
                title="Embedding Generation Results",
                total=len(embedded_df),
                success=len(embedded_df),
                failure=0,
            )
    # -------------------
    # temporary button
    # -------------------

    # if st.button("Initialize Database"):
    #     initialize_database()
    #     st.success("Database initialized.")

    # ---------------------------------------------------------------------
    # Topic Modeling
    # ---------------------------------------------------------------------

    st.write("Topic Modeling")

    col1, col2 = st.columns(2)

    # with col1:

    #     if st.button(
    #         "Run Topic Modeling",
    #         use_container_width=True
    #     ):

    #         try:

    #             topic_count = run_topic_model()

    #             st.success(
    #                 f"Topic modeling completed successfully.\n\n"
    #                 f"{topic_count} topic(s) generated."
    #             )

    #         except Exception as e:

    #             st.error(str(e))

    with col1:      # Temporary change to avoid conflict with the `st` import in topic_model.py
        if st.button("Run Topic Modeling", use_container_width=True):
            try:
                topic_count = run_topic_model()
                st.success(f"Topic modeling completed successfully.\n\n{topic_count} topic(s) generated.")
            except Exception as e:
                st.error(f"Exception: {e}")
                st.exception(e)  # full traceback in the UI, not just the message

    # -------------------
    # Year-specific Topic Modeling
    # -------------------

    # st.subheader("Yearly Topic Analysis")
    st.write("Yearly Topic Modeling")
    col1, col2 = st.columns(2)
       

    with col1:

        years = get_available_years()

        if not years:
            st.info("No papers with a year found in the database yet.")
        else:
            selected_year = st.selectbox("Select a year", years)

            if st.button("Run Yearly Topic Modeling"):
                try:
                    with st.spinner(f"Running topic modeling for {selected_year}..."):
                        topics_df = run_topic_model_for_year(selected_year)

                    st.success(
                                f"Topic modeling completed for {selected_year}. "
                                f"{len(topics_df)} topic(s) generated."
                            )

                    st.dataframe(
                        topics_df.rename(columns={
                            "topic_id": "Topic ID",
                            "topic_name": "Topic Name",
                            "keywords": "Keywords",
                            "paper_count": "Paper Count",
                        })[["Topic ID", "Topic Name", "Keywords", "Paper Count"]],
                        width="stretch",
                    )                    

                except ValueError as e:
                    st.warning(str(e))
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.exception(e)

    


    # ---------------------------------------------------------------------
    # Navigation to Analysis Pages
    # ---------------------------------------------------------------------
    st.write("Navigation to Analysis Pages")


    col1, col2 = st.columns(2)

    # ---------------------------------------------------------------------
    # Trend Analysis Page button
    # ---------------------------------------------------------------------

    with col1:
        #pages\topic_trend_analysis.py

        # if st.button(
        #         "📈 Go to Trend Analysis → ",
        #         use_container_width=True
        #     ):
        
        #         st.session_state.page = "trend"
        #         st.rerun()

        st.page_link(
                    "pages/topic_trend_analysis.py",
                    label="📈 Go to Topic Trend Analysis →",
                    use_container_width=True
                )
    # -------------------
    # Topic Evolution Analysis Page button
    # -------------------

    with col2:

        st.page_link(
            "pages/topic_evolution.py",
            label="📈 Go to Topic Evolution Analysis →",
            use_container_width=True
        )


# ------------------------------------------------------
# Page for Trend Analysis
# ------------------------------------------------------

# ---------------------------------------------------------------------
# Trend Analysis Interface - moved to pages/topic_trend_analysis.py
# ---------------------------------------------------------------------

