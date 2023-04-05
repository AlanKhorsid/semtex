import re
import json
from numpy import empty
import os
import html
from decouple import config

# from bingsearchapi import call_manually
from bestmatch import get_best_title_match
from preprocesschecker import check_spellchecker_threaded
from pathlib import Path

from util import pickle_load, pickle_save

rootpath = str(Path(__file__).parent.parent.parent)
src_folder = f"{rootpath}/datasets/BingSearchResults"
all_search_results = pickle_load("searchres", is_dump=True)

num_of_missed_queries = 0


def generate_suggestion(query):
    if not query in all_search_results:
        num_of_missed_queries += 1
        print(f"NOT FOUND: {query}")
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
