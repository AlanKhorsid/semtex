import re
import json
from numpy import empty
import os
import html
from decouple import config
from bestmatch import get_best_title_match
from preprocesschecker import check_spellchecker_threaded

src_folder = "/Users/alankhorsid/Documents/semtex/datasets/BingSearchResults"


def search_for_JSON(query_string):
    """
    Searches for a JSON file containing the search results for a given query.
    If the file is found, the JSON object is returned.
    If the file is not found, the original query is returned.

    Args:
        query_string (str): The search query.

    Returns:
        dict: The JSON object containing the search results for the given query.

    Example:
        >>> search_for_JSON("Barack Obama")
        {
            "_type": "SearchResponse",
            "queryContext": {
                "originalQuery": "Barack Obama"
            },  ...
    """

    for filename in os.listdir(src_folder):
        if filename.endswith(".json"):
            filepath = os.path.join(src_folder, filename)
            with open(filepath, "r") as f:
                json_data = json.load(f)
                if (
                    json_data["_type"] == "SearchResponse"
                    and json_data["queryContext"]["originalQuery"].lower()
                    == query_string.lower()
                ):
                    return json_data
    return query_string


def generate_suggestion(query):
    """
        Generates a suggested alternative search query based on the search results for the original query.
        If the original query is not found in the JSON file, the original query is returned.
        If the original query is found in the JSON file,
        but no suggested alternative query is given,
        the best match for the original query based on search result titles is returned.
        If the original query is found in the JSON file,
        and a suggested alternative query is given, the suggested alternative query is returned.
    Args:
        query (str): The original search query.

    Returns:
        str: The suggested alternative search query, or the best match for the original query based on search result titles.

    Example:
        >>> generate_suggestion("Barak Obma")
        "Barack Obama"
    """

    try:
        json_obj = search_for_JSON(query)
        # print(f"Found query in JSON file: {query}")
    except FileNotFoundError:
        print(f"File not found for: {query}")
        return query

    if not "webPages" in json_obj:
        return query
    elif "alteredQuery" in json_obj["queryContext"]:
        # get the suggested query given by Bing
        suggestion = json_obj["queryContext"]["alteredQuery"]
        # check if the suggestion is contained in any of the titles, if so, return the suggestion
        if "value" in json_obj["webPages"]:
            results = json_obj["webPages"]["value"]
            titles = [result["name"] for result in results]
            for title in titles:
                if query in title:
                    return html.unescape(suggestion)
        return html.unescape(get_best_title_match(query, titles))
    elif "value" in json_obj["webPages"]:
        # get all search result titles
        results = json_obj["webPages"]["value"]
        titles = [result["name"] for result in results]
        # get the best match for the original query based on the titles
        suggestion = get_best_title_match(query, titles)
        return html.unescape(suggestion)


# check_spellchecker_threaded(generate_suggestion, num_threads=100, only_hard=True)
