from pathlib import Path
import sqlite3
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATABASE_PATH = PROJECT_ROOT / "data" / "metadata.db"

# ------------------------------------------------------------------
# Initialize the database and create tables if they don't exist
# ------------------------------------------------------------------

def initialize_database():

    conn = sqlite3.connect(DATABASE_PATH)

    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS papers(

        source_file TEXT PRIMARY KEY,

        title TEXT,

        authors TEXT,

        abstract TEXT,

        keywords TEXT,

        year INTEGER,

        document TEXT

    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS year_topics(

        year INTEGER,

        topic_id INTEGER,

        topic_name TEXT,

        keywords TEXT,

        paper_count INTEGER,

        created_at TEXT,

        PRIMARY KEY (year, topic_id)

    )
    """)

    cursor.execute("PRAGMA table_info(papers)")
    columns = [column[1] for column in cursor.fetchall()]

    if "embedded" not in columns:
        cursor.execute("""
            ALTER TABLE papers
            ADD COLUMN embedded INTEGER DEFAULT 0
    """)

    cursor.execute("PRAGMA table_info(papers)")
    columns = [column[1] for column in cursor.fetchall()]

    if "topic_id" not in columns:
        cursor.execute("""
            ALTER TABLE papers
            ADD COLUMN topic_id INTEGER
        """)

    conn.commit()
    conn.close()

# ------------------------------------------------------------------
# Append metadata to the database
# ------------------------------------------------------------------

def append_metadata(meta_df: pd.DataFrame):

    initialize_database()

    conn = sqlite3.connect(DATABASE_PATH)

    meta_df = meta_df.copy()

    meta_df["authors"] = meta_df["authors"].apply(
    lambda x: ", ".join(x) if isinstance(x, list) else ""
    )

    meta_df["keywords"] = meta_df["keywords"].apply(
        lambda x: ", ".join(x) if isinstance(x, list) else ""
    )

    cursor = conn.cursor()

    for _, row in meta_df.iterrows():

        cursor.execute(
            """
            INSERT OR IGNORE INTO papers
            (
                source_file,
                title,
                authors,
                abstract,
                keywords,
                year,
                document
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["source_file"],
                row["title"],
                row["authors"],
                row["abstract"],
                row["keywords"],
                row["year"],
                row["document"]
            )
        )

    conn.commit()
    
    conn.close()

# ------------------------------------------------------------------
# Get all metadata from the database
# ------------------------------------------------------------------

def load_metadata():

    initialize_database()

    conn = sqlite3.connect(DATABASE_PATH)

    df = pd.read_sql("SELECT * FROM papers", conn)

    conn.close()

    return df

# ------------------------------------------------------------------
# Get available years from the database
# ------------------------------------------------------------------

def get_available_years():

    initialize_database()

    conn = sqlite3.connect(DATABASE_PATH)

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT DISTINCT year
        FROM papers
        WHERE year IS NOT NULL
        ORDER BY year
        """
    )

    years = [
        row[0]
        for row in cursor.fetchall()
    ]

    conn.close()

    return years

# ------------------------------------------------------------------
# Load papers by year from the database
# ------------------------------------------------------------------

def load_papers_by_year(year):

    initialize_database()

    conn = sqlite3.connect(DATABASE_PATH)

    df = pd.read_sql(
        """
        SELECT
            source_file,
            title,
            authors,
            abstract,
            keywords,
            year,
            document
        FROM papers
        WHERE year = ?
        """,
        conn,
        params=(year,)
    )

    conn.close()

    return df

# ------------------------------------------------------------------
# Save year topics to the database, replace when rerunning topic modeling for the same year
# ------------------------------------------------------------------

def save_year_topics(year, topics_df):

    initialize_database()

    conn = sqlite3.connect(DATABASE_PATH)

    cursor = conn.cursor()

    cursor.execute(
        """
        DELETE FROM year_topics
        WHERE year = ?
        """,
        (year,)
    )

    topics_df.to_sql(
        "year_topics",
        conn,
        if_exists="append",
        index=False
    )

    conn.commit()

    conn.close()

# ------------------------------------------------------------------
# Load year topics from the database
# ------------------------------------------------------------------

def load_year_topics(year):

    initialize_database()

    conn = sqlite3.connect(DATABASE_PATH)

    df = pd.read_sql(
        """
        SELECT *
        FROM year_topics
        WHERE year = ?
        ORDER BY topic_id
        """,
        conn,
        params=(year,)
    )

    conn.close()

    return df