import pandas as pd

from src.database import load_year_topics

from .similarity import jaccard_similarity


def compare_topics(
    previous_year,
    current_year,
    threshold=0.3
):
    """
    Compare topics between two consecutive years.

    Each topic from the previous year is treated as a
    candidate theme and compared with topics from the
    following year.

    Possible statuses:

        Growing
        Declining
        Persisted
        Disappeared
        New
    """

    # --------------------------------------------------
    # Load yearly topic models
    # --------------------------------------------------

    previous_topics = load_year_topics(
        previous_year
    )

    current_topics = load_year_topics(
        current_year
    )

    if previous_topics.empty:
        return pd.DataFrame()

    if current_topics.empty:
        return pd.DataFrame()

    # --------------------------------------------------
    # Store comparison results
    # --------------------------------------------------

    results = []

    # Keep track of current-year topics that have already
    # been matched to a previous-year topic.
    matched_current_topics = set()

    # --------------------------------------------------
    # Compare every previous-year topic
    # --------------------------------------------------

    for _, old_topic in previous_topics.iterrows():

        best_similarity = 0.0
        best_match = None

        for index, new_topic in current_topics.iterrows():

            # Do not match the same current topic to
            # multiple previous topics.
            if index in matched_current_topics:
                continue

            similarity = jaccard_similarity(
                old_topic["keywords"],
                new_topic["keywords"]
            )

            if similarity > best_similarity:

                best_similarity = similarity
                best_match = index

        # --------------------------------------------------
        # A matching topic was found
        # --------------------------------------------------

        if (
            best_match is not None
            and best_similarity >= threshold
        ):

            new_topic = current_topics.loc[
                best_match
            ]

            matched_current_topics.add(
                best_match
            )

            old_count = int(
                old_topic["paper_count"]
            )

            new_count = int(
                new_topic["paper_count"]
            )

            # ----------------------------------------------
            # Determine evolution status
            # ----------------------------------------------

            if new_count > old_count:

                status = "Growing"

            elif new_count < old_count:

                status = "Declining"

            else:

                status = "Persisted"

            results.append({

                "previous_topic":
                    old_topic["topic_name"],

                "current_topic":
                    new_topic["topic_name"],

                "similarity":
                    round(
                        best_similarity,
                        3
                    ),

                "previous_papers":
                    old_count,

                "current_papers":
                    new_count,

                "status":
                    status
            })

        # --------------------------------------------------
        # No sufficiently similar topic in next year
        # --------------------------------------------------

        else:

            results.append({

                "previous_topic":
                    old_topic["topic_name"],

                "current_topic":
                    None,

                "similarity":
                    round(
                        best_similarity,
                        3
                    ),

                "previous_papers":
                    int(old_topic["paper_count"]),

                "current_papers":
                    0,

                "status":
                    "Disappeared"
            })

    # --------------------------------------------------
    # Identify genuinely new topics in current year
    # --------------------------------------------------

    for index, new_topic in current_topics.iterrows():

        if index not in matched_current_topics:

            results.append({

                "previous_topic":
                    None,

                "current_topic":
                    new_topic["topic_name"],

                "similarity":
                    0.0,

                "previous_papers":
                    0,

                "current_papers":
                    int(new_topic["paper_count"]),

                "status":
                    "New"
            })

    # --------------------------------------------------
    # Return results
    # --------------------------------------------------

    return pd.DataFrame(results)