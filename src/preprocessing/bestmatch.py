import Levenshtein


def generate_title_permutations(title):
    """Generate permutations of a given title.

    Args:
        title (str): The title to generate permutations for.

    Returns:
        A list of title permutations.
    """
    title = title.split()
    n = len(title)
    result = [
        " ".join([title[j] for j in range(n) if i & (1 << j)]) for i in range(1, 2**n)
    ]
    result = [x for x in result if len(x.split()) > 1]
    result.extend(title)
    permutations = result if len(title) > 0 and len(result) > 0 else []
    return permutations


def compare_title_permutations_with_query(title, query):
    """Compare title permutations with a given query.

    Args:
        title (str): The title to compare permutations for.
        query (str): The query to compare against.

    Returns:
        A list of tuples, where each tuple contains a title permutation and its Levenshtein distance from the query.
    """
    permutations = generate_title_permutations(title)
    results = []
    for p in permutations:
        distance = 1 - Levenshtein.ratio(query.lower(), p.lower())
        results.append((p, distance))
    return results


def get_best_title_match(query, titles):
    """Get the best title match for a given query.

    Args:
        query (str): The query to match against.
        titles (list): A list of titles to search for matches.

    Returns:
        The best matching title, or None if no matches were found.
    """
    best_match = None
    best_distance = float("inf")
    for title in titles:
        results = compare_title_permutations_with_query(title, query)
        for r in results:
            if r[1] < best_distance:
                best_distance = r[1]
                best_match = r[0]
    return best_match
