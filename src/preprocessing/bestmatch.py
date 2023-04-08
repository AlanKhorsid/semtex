import Levenshtein


def generate_title_permutations(title):
    # remove " - Wikidata" from title
    title = title.replace(" - Wikidata", "")
    title_words = title.split()
    if len(title_words) > 20:
        print(f"Title too long: {title}")
        return [title]
    result = []
    for i in range(1, 2 ** len(title_words)):
        perm = " ".join([title_words[j] for j in range(len(title_words)) if i & (1 << j)])
        if perm != title and len(perm.split()) > 0:
            result.append(perm)
    result.append(title)
    return result


def compare_title_permutations_with_query(title, query):
    permutations = generate_title_permutations(title)
    return [(p, Levenshtein.ratio(query.lower(), p.lower())) for p in permutations]


def remove_last_symbol(best_match):
    symbols = [",", ":", ";", "-", ".", " ", "?", "!"]
    while len(best_match) > 0 and best_match[-1] in symbols:
        best_match = best_match[:-1]
    return best_match


def get_best_title_match(query, titles):
    best_match = None
    highest_score = float("-inf")
    for title in titles:
        results = compare_title_permutations_with_query(title, query)
        for r in results:
            if r[1] == 1.0:
                return query
            if r[1] > highest_score:
                highest_score = r[1]
                best_match = r[0]
    if best_match is None or not is_acceptable_match(best_match):
        return query
    if highest_score <= 0.8875:
        return query
    best_match = remove_last_symbol(best_match)
    best_match = best_match.replace("\u2019", "'")
    return best_match


def is_acceptable_match(suggestion):
    return not suggestion.endswith("...")
