import sqlite3
from itertools import combinations
from collections import Counter

import pandas as pd
import networkx as nx
import plotly.graph_objects as go

from src.database import DATABASE_PATH


def get_year_keyword_data(year, top_n=12):
    conn = sqlite3.connect(DATABASE_PATH)
    df = pd.read_sql(
        "SELECT source_file, keywords FROM papers WHERE year = ?",
        conn, params=(year,)
    )
    conn.close()

    paper_keywords = [
        [k.strip().lower() for k in str(kw).split(",") if k.strip()]
        for kw in df["keywords"]
    ]

    counts = Counter(k for kws in paper_keywords for k in kws)
    top_keywords = {k for k, _ in counts.most_common(top_n)}

    cooccurrence = Counter()
    for kws in paper_keywords:
        present = [k for k in kws if k in top_keywords]
        for a, b in combinations(sorted(set(present)), 2):
            cooccurrence[(a, b)] += 1

    return counts, top_keywords, cooccurrence


def build_keyword_bubble_chart(year, top_n=20):
    counts, top_keywords, cooccurrence = get_year_keyword_data(year, top_n)

    G = nx.Graph()
    for kw in top_keywords:
        G.add_node(kw, count=counts[kw])
    for (a, b), weight in cooccurrence.items():
        G.add_edge(a, b, weight=weight)

    pos = nx.spring_layout(G, weight="weight", seed=42, k=0.3)

    nodes = list(G.nodes())
    x = [pos[n][0] for n in nodes]
    y = [pos[n][1] for n in nodes]
    sizes = [max(counts[n] * 40, 25) for n in nodes]
    labels = [f"{n} ({counts[n]})" for n in nodes]

    fig = go.Figure(data=[go.Scatter(
        x=x, y=y,
        mode="markers+text",
        marker=dict(size=sizes, sizemode="area", opacity=0.6),
        text=labels,
        textposition="middle center",
        textfont=dict(size=14),
        hovertext=[f"{n}: {counts[n]} paper(s)" for n in nodes],
        hoverinfo="text",
    )])

    fig.update_layout(
        title=f"Keyword Landscape — {year}",
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=500,
    )

    return fig