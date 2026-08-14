import pandas as pd


def generate_trend(papers_df, full_year_range=True):
    counts = (
        papers_df
        .groupby("year")
        .size()
        .reset_index(name="count")
    )

    if full_year_range and not counts.empty:
        all_years = pd.DataFrame({
            "year": range(int(counts["year"].min()), int(counts["year"].max()) + 1)
        })
        counts = all_years.merge(counts, on="year", how="left").fillna({"count": 0})

    counts["count"] = counts["count"].astype(int)

    titles_by_year = (
        papers_df.groupby("year")["title"]
        .apply(lambda titles: "<br>".join(f"• {t}" for t in titles))
        .reset_index(name="titles")
    )

    trend = counts.merge(titles_by_year, on="year", how="left")
    trend["titles"] = trend["titles"].fillna("No related papers")

    return trend.sort_values("year")


def generate_topic_distribution(papers_df, topic_labels_df):
    total = len(papers_df)

    dist = (
        papers_df[papers_df["topic_id"] != -1]
        .groupby("topic_id")
        .size()
        .reset_index(name="count")
    )
    dist = dist.merge(topic_labels_df, on="topic_id", how="left")
    dist["label"] = dist["topic_name"].fillna("Topic " + dist["topic_id"].astype(str))
    dist["percentage"] = (dist["count"] / total * 100).round(1)

    return dist.sort_values("count", ascending=False)