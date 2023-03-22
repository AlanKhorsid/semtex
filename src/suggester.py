import re
import json
from numpy import empty
import os
import html
from decouple import config
from bestmatch import get_best_title_match
from preprocessing.preprocesschecker import (
    check_spellchecker,
    check_spellchecker_threaded,
)
from pathlib import Path

from util import pickle_load, pickle_save

rootpath = str(Path(__file__).parent.parent.parent)
src_folder = "/Users/alankhorsid/Documents/semtex/datasets/BingSearchResults"


def search_for_JSON(query):
    for filename in os.listdir(src_folder):
        if filename.endswith(".json"):
            filepath = os.path.join(src_folder, filename)
            with open(filepath, "r") as f:
                json_data = json.load(f)
                if (
                    json_data["_type"] == "SearchResponse"
                    and json_data["queryContext"]["originalQuery"].lower()
                    == query.lower()
                ):
                    return json_data
    return query


def generate_suggestion(query):
    # all_search_results = pickle_load("all-test-cells-search-results", is_dump=True)
    # json_obj = all_search_results[query]

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
                if suggestion in title:
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


check_spellchecker(generate_suggestion, only_hard=True)
