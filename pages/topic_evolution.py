from src.topic_evolution.transition_evaluation import evaluate_transition
from src.database import get_available_years
import streamlit as st

st.subheader("Year-to-Year Topic Evolution")

years = get_available_years()

col_a, col_b = st.columns(2)
with col_a:
    year_from = st.selectbox("From year", years, key="evo_from")
with col_b:
    year_to = st.selectbox("To year", years, index=min(1, len(years)-1), key="evo_to")

if st.button("Run Evolution Analysis"):
    if year_from == year_to:
        st.warning("Choose two different years.")
    else:
        try:
            with st.spinner(f"Fitting {year_from} model and transforming {year_to} papers..."):
                result = evaluate_transition(year_from, year_to)

            st.success(f"Evolution analysis complete: {year_from} → {year_to}")

            st.dataframe(result["evolution_df"], width="stretch")

            st.subheader(f"Papers in {year_to} not matching any {year_from} topic")
            if result["outlier_papers"].empty:
                st.write("None — all papers fit an existing topic.")
            else:
                st.write(f"{len(result['outlier_papers'])} paper(s)")
                if result["emerging_keywords"]:
                    st.write(f"**Candidate emerging theme keywords:** {result['emerging_keywords']}")
                st.dataframe(result["outlier_papers"][["title", "authors", "year"]], width="stretch")

        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)
