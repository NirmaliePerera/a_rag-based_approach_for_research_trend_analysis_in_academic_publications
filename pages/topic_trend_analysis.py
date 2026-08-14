import streamlit as st
from pathlib import Path
import plotly.express as px

from src.retriever import get_topic_labels, retrieve_papers_by_threshold #, retrieve_papers
from src.trend_analysis import generate_topic_distribution, generate_trend
from src.summarize import generate_summary

st.set_page_config(layout="wide")       # default was layout="centered" ( making it narrow, ~730px)

st.title(
        "Research Trend Analysis"
    )


query = st.text_input(
    "Enter research topic/query"
)


if st.button("Analyze"):

    if not query:

        st.warning(
            "Please enter a query."
        )

    else:

        papers_df = retrieve_papers_by_threshold(query, max_distance=1.2)


        if papers_df.empty:

            st.warning(
                "No related papers found."
            )


        else:

            st.success(
                f"{len(papers_df)} related papers found."
            )


            # -------------------------
            # Trend chart
            # -------------------------

            # st.subheader(
            #     "Publication Trend"
            # )

            topic_labels = get_topic_labels()

            trend_df = generate_trend(papers_df)
            topic_df = generate_topic_distribution(papers_df, topic_labels)

            row1_col1, row1_col2 = st.columns(2)

            with row1_col1:
                st.subheader("Publication Trend")
                fig1 = px.line(
                    trend_df,
                    x="year",
                    y="count",
                    markers=True,
                    text="count",
                    custom_data=["titles"],
                )

                fig1.update_traces(
                    textposition="top center",
                    hovertemplate="Year: %{x}<br>Papers: %{y}<br><br>%{customdata[0]}<extra></extra>",
                )

                fig1.update_xaxes(type="category")
                fig1.update_yaxes(title="Number of related papers")
                fig1.update_layout(height=350)
                st.plotly_chart(fig1, width="stretch")

            with row1_col2:

                st.subheader("Topic Distribution (Retrieved Papers)")

                fig2 = px.bar(
                    topic_df,
                    x="label",          # was "topic_id"
                    y="percentage",     # was "count"
                    text="percentage",
                    hover_data=["count"],
                )

                fig2.update_traces(texttemplate="%{text}%")
                fig2.update_xaxes(type="category", title="Topic")
                fig2.update_yaxes(title="% of retrieved papers")
                fig2.update_layout(bargap=0.4, height=350)
                st.plotly_chart(fig2, width="stretch")

            row2_col1, row2_col2 = st.columns(2)

            with row2_col1:
                st.subheader("Trend Summary")

                with st.spinner("Generating summary..."):
                    summary = generate_summary(papers_df, query)

                st.write(summary)

            with row2_col2:
                st.subheader("Related Papers")

                display_df = papers_df[["title", "authors", "year"]].reset_index(drop=True)
                display_df.index = display_df.index + 1
                display_df.index.name = "Paper ID"

                st.dataframe(display_df)

            # trend_df = generate_trend(
            #     papers_df
            # )


            # st.bar_chart(
            #     trend_df.set_index("year")
            # )


            # # -------------------------
            # # Summary + papers
            # # -------------------------

            # col1, col2 = st.columns(2)


            # with col1:

            #     st.subheader(
            #         "Summary"
            #     )


            #     summary = generate_summary(
            #         papers_df
            #     )


            #     st.write(summary)



            # with col2:

            #     st.subheader(
            #         "Related Papers"
            #     )


            #     display_df = papers_df[
            #         [
            #             "title",
            #             "authors",
            #             "year"
            #         ]
            #     ]


            #     st.dataframe(
            #         display_df,
            #         use_container_width=True
            #     )


    if st.button("Back"):

        st.session_state.page = "main"
        st.rerun()