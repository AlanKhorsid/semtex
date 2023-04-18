import re
import json
from numpy import empty
import os
import html
from decouple import config

# from bingsearchapi import call_manually
from .bestmatch import get_best_title_match
from .preprocesschecker import check_spellchecker
from pathlib import Path

from util import pickle_load, pickle_save

rootpath = str(Path(__file__).parent.parent.parent)
src_folder = f"{rootpath}/datasets/BingSearchResults"


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


all_search_results = pickle_load("all-validation-cells-bing", is_dump=True)


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

    if not query in all_search_results:
        return query
    json_obj = all_search_results[query]

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

    else:
        # get the best match for the original query based on search result titles
        if "value" in json_obj["webPages"]:
            results = json_obj["webPages"]["value"]
            titles = [result["name"] for result in results]
            return html.unescape(get_best_title_match(query, titles))
        else:
            return query


# check_spellchecker(generate_suggestion, only_hard=True, case_sensitive=True)
