import Levenshtein


def generate_title_permutations(title):
    """
    Generate permutations of a given title.

    Args:
        title (str): The title to generate permutations for.

    Returns:
        A list of title permutations.

    Example:
        >>> generate_title_permutations("I love eating")
        ["I love eating", "I love", "I eating", "love eating", "I", "love", "eating"]
    """
    if len(title.split()) > 20:
        print(f"Title too long: {title}")
        return [title]
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
    """
    Compare title permutations with a given query.

    Args:
        title (str): The title to compare permutations for.
        query (str): The query to compare against.

    Returns:
        A list of tuples, where each tuple contains a title permutation and its Levenshtein distance from the query.

    Example:
        >>> compare_title_permutations_with_query("I love python", "love python")
        [('love python', 0.0), ('I love python', 0.25), ('I love', 0.5), ('I python', 0.5), ('I', 0.75), ('love', 0.75), ('python', 0.75)]

    """

    permutations = generate_title_permutations(title)
    results = []
    for p in permutations:
        distance = 1 - Levenshtein.ratio(query.lower(), p.lower())
        results.append((p, distance))
    return results


def remove_symbol(best_match, symbol=","):
    if best_match[-1] == symbol:
        best_match = best_match[:-1]
    return best_match


def get_best_title_match(query, titles):
    """
    Get the best matching title from a list of titles. The best matching title is the title with the lowest Levenshtein distance from the query to any of its permutations.

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
    best_match = remove_symbol(best_match, symbol=",")
    best_match = remove_symbol(best_match, symbol=":")
    best_match = remove_symbol(best_match, symbol=";")
    best_match = remove_symbol(best_match, symbol="-")

    if not is_acceptable_match(best_match):
        return query
    return best_match


# If the best match has "..." at the end, then the query is the best match
def is_acceptable_match(suggestion):
    if suggestion[-3:] == "...":
        return False
    else:
        return True
