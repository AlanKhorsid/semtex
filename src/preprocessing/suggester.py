import requests
from decouple import config
from bestmatch import get_best_title_match
from preprocesschecker import check_spellchecker_threaded

subscription_key = config("subscription_key", default="")
search_url = "https://api.bing.microsoft.com/v7.0/search"
headers = {"Ocp-Apim-Subscription-Key": subscription_key}


def generate_suggestion(query):
    """
    Preprocesses a search query by checking for spelling errors and suggesting alternatives.

    Args:
        query (str): The original search query.

    Returns:
        str: The suggested alternative search query, or the best match for the original query based on search result titles.

    """
    params = {"q": query, "textDecorations": True, "textFormat": "HTML"}
    try:
        response = requests.get(search_url, headers=headers, params=params)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        # print(f"HTTP error occurred: {e}")
        return None

    search_results = response.json()
    try:
        suggestion = search_results["queryContext"]["alteredQuery"]
        return suggestion
    except:
        results = search_results["webPages"]["value"]
        titles = [result["name"] for result in results]
        suggestion = get_best_title_match(query, titles)
        # print(f"Found best match: {suggestion}")
        return suggestion


# check_spellchecker_threaded(generate_suggestion, num_threads=100)
