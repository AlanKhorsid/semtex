import json
import os
import html
import pickle
from typing import Literal

# from bingsearchapi import call_manually
from .bestmatch import get_best_title_match
from .preprocesschecker import check_spellchecker_threaded
from pathlib import Path


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
                    and json_data["queryContext"]["originalQuery"].lower() == query_string.lower()
                ):
                    return json_data
    return query_string


def pickle_load(filename, is_dump: bool = False):
    ROOTPATH = Path(__file__).parent.parent.parent
    file = f"{ROOTPATH}/src/{'pickle-dumps' if is_dump else 'pickles'}/{filename}.pickle"
    with open(file, "rb") as f:
        return pickle.load(f)


all_search_results = None


def generate_suggestion(query, dataset: Literal["test", "validation"], year: Literal["2022", "2023"]):
    global all_search_results
    if all_search_results is None:
        if year == "2022":
            all_search_results = pickle_load("bingsearches", is_dump=True)
        elif year == "2023":
            all_search_results = pickle_load(f"bing-results-{dataset}-2023", is_dump=True)

    query = query.strip()
    assert query in all_search_results, f"Query {query} not found in Bing search results."

    json_obj = all_search_results[query]
    assert json_obj is not None, f"JSON object for query {query} is None."

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


def release_search_results():
    global all_search_results
    all_search_results = None
