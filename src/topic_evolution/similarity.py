def keyword_set(keyword_string):
    """
    Convert a comma-separated keyword string into
    a normalized set of keywords.
    """

    if not keyword_string:
        return set()

    return {
        keyword.strip().lower()
        for keyword in keyword_string.split(",")
        if keyword.strip()
    }


def jaccard_similarity(keywords_a, keywords_b):
    """
    Calculate Jaccard similarity between two keyword sets.

    J(A,B) = |A intersection B| / |A union B|
    """

    set_a = keyword_set(keywords_a)
    set_b = keyword_set(keywords_b)

    if not set_a or not set_b:
        return 0.0

    intersection = len(
        set_a.intersection(set_b)
    )

    union = len(
        set_a.union(set_b)
    )

    return intersection / union