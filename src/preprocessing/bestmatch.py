import html
import random
import itertools
from Levenshtein import ratio


def generate_title_combinations(title: str) -> list:
    title = title.replace(" - Wikidata", "")
    title = title.replace(" — Wikidata", "")
    words = title.split()
    if len(words) > 4:
        print(f"Title has too many words: {html.unescape(title)}")
        return [title]

    all_combinations = set()
    for length in range(1, len(words) + 1):
        for combination in itertools.combinations(words, length):
            all_combinations.add(" ".join(combination))
    all_combinations.add(title)
    return list(all_combinations)


def compare_title_combinations_with_query(title: str, query: str) -> list:
    combinations = generate_title_combinations(title)
    return [(p, 1 - ratio(query.lower(), p.lower())) for p in combinations]


x = compare_title_combinations_with_query("Tesla Model 3", "Tesla Model 3 car")
print(1 - ratio("Tesla Model 3 car", "Tesla Model 3"))


def get_best_title_match(
    query: str, titles: list, distance_tolerance: float = 0.12
) -> str:
    best_match = None
    lowest_distance = float("inf")
    for title in titles:
        results = compare_title_combinations_with_query(title, query)
        for r in results:
            if r[1] < lowest_distance:
                lowest_distance = r[1]
                best_match = r[0]

    if (
        lowest_distance > distance_tolerance
        or best_match is None
        or is_too_long_title(best_match)
    ):
        return query

    best_match = remove_last_symbol(best_match)
    best_match = best_match.replace("\u2019", "'")
    return best_match


def remove_last_symbol(best_match: str) -> str:
    symbols = [",", ":", ";", "-", ".", " ", "?", "!"]
    while len(best_match) > 0 and best_match[-1] in symbols:
        best_match = best_match[:-1]
    return best_match


def is_too_long_title(suggestion: str) -> bool:
    return suggestion.endswith("...")
