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




# import streamlit as st

# from src.database import get_available_years
# from src.topic_evolution.evolution import compare_topics


# # --------------------------------------------------
# # Page configuration
# # --------------------------------------------------

# st.set_page_config(
#     page_title="Topic Evolution Analysis",
#     layout="wide"
# )


# st.title("Topic Evolution Analysis")

# st.write(
#     "Compare research topics between two consecutive years "
#     "to identify emerging, growing, declining, and disappearing themes."
# )


# # --------------------------------------------------
# # Get available years
# # --------------------------------------------------

# years = get_available_years()


# if len(years) < 2:

#     st.warning(
#         "At least two years of topic models are required "
#         "for topic evolution analysis."
#     )

#     st.stop()


# # --------------------------------------------------
# # Select base year
# # --------------------------------------------------

# available_base_years = [
#     year for year in years
#     if year + 1 in years
# ]


# if not available_base_years:

#     st.warning(
#         "No consecutive years are available for comparison."
#     )

#     st.stop()


# selected_year = st.selectbox(
#     "Select base year",
#     available_base_years
# )


# next_year = selected_year + 1


# st.info(
#     f"Topic evolution will be evaluated from "
#     f"**{selected_year} → {next_year}**."
# )


# # --------------------------------------------------
# # Run topic evolution analysis
# # --------------------------------------------------

# if st.button(
#     "Compare Topic Evolution",
#     use_container_width=True
# ):

#     try:

#         with st.spinner(
#             f"Comparing topics from {selected_year} "
#             f"with {next_year}..."
#         ):

#             results = compare_topics(
#                 selected_year,
#                 next_year
#             )


#         # --------------------------------------------------
#         # No results
#         # --------------------------------------------------

#         if results.empty:

#             st.warning(
#                 "No topic comparison results were generated."
#             )

#             st.stop()


#         # --------------------------------------------------
#         # Results table
#         # --------------------------------------------------

#         st.subheader(
#             f"Topic Evolution: {selected_year} → {next_year}"
#         )

#         st.dataframe(
#             results,
#             use_container_width=True
#         )


#         # --------------------------------------------------
#         # Evolution status chart
#         # --------------------------------------------------

#         st.subheader("Evolution Status")

#         status_count = (
#             results["status"]
#             .value_counts()
#         )

#         st.bar_chart(
#             status_count
#         )


#         # --------------------------------------------------
#         # Topic growth comparison
#         # --------------------------------------------------

#         st.subheader(
#             "Topic Growth Comparison"
#         )

#         growth_df = results[
#             results["current_topic"].notna()
#         ].copy()


#         if not growth_df.empty:

#             chart_df = growth_df[
#                 [
#                     "current_topic",
#                     "previous_papers",
#                     "current_papers"
#                 ]
#             ].copy()


#             chart_df = chart_df.set_index(
#                 "current_topic"
#             )


#             st.bar_chart(
#                 chart_df
#             )

#         else:

#             st.info(
#                 "No continuing topics were identified."
#             )


#         # --------------------------------------------------
#         # Emerging themes
#         # --------------------------------------------------

#         st.subheader(
#             "Emerging Themes"
#         )

#         emerging = results[
#             results["status"] == "New"
#         ]


#         if not emerging.empty:

#             st.dataframe(
#                 emerging[
#                     [
#                         "current_topic",
#                         "current_papers"
#                     ]
#                 ],
#                 use_container_width=True
#             )

#         else:

#             st.info(
#                 "No new topics were detected."
#             )


#         # --------------------------------------------------
#         # Growing topics
#         # --------------------------------------------------

#         st.subheader(
#             "Growing Topics"
#         )

#         growing = results[
#             results["status"] == "Growing"
#         ]


#         if not growing.empty:

#             st.dataframe(
#                 growing[
#                     [
#                         "previous_topic",
#                         "current_topic",
#                         "previous_papers",
#                         "current_papers"
#                     ]
#                 ],
#                 use_container_width=True
#             )

#         else:

#             st.info(
#                 "No growing topics were detected."
#             )


#         # --------------------------------------------------
#         # Declining topics
#         # --------------------------------------------------

#         st.subheader(
#             "Declining Topics"
#         )

#         declining = results[
#             results["status"] == "Declining"
#         ]


#         if not declining.empty:

#             st.dataframe(
#                 declining[
#                     [
#                         "previous_topic",
#                         "current_topic",
#                         "previous_papers",
#                         "current_papers"
#                     ]
#                 ],
#                 use_container_width=True
#             )

#         else:

#             st.info(
#                 "No declining topics were detected."
#             )


#         # --------------------------------------------------
#         # Disappeared topics
#         # --------------------------------------------------

#         st.subheader(
#             "Disappeared Topics"
#         )

#         disappeared = results[
#             results["status"] == "Disappeared"
#         ]


#         if not disappeared.empty:

#             st.dataframe(
#                 disappeared[
#                     [
#                         "previous_topic",
#                         "previous_papers"
#                     ]
#                 ],
#                 use_container_width=True
#             )

#         else:

#             st.info(
#                 "No disappeared topics were detected."
#             )


#     except Exception as e:

#         st.error(
#             f"Topic evolution analysis failed: {e}"
#         )