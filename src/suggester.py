import json
import os
import html
from pathlib import Path
from numpy import empty
from decouple import config

from preprocessing.bestmatch import get_best_title_match
from preprocessing.preprocesschecker import check_spellchecker
from util import pickle_load, pickle_save

rootpath = str(Path(__file__).parent.parent.parent)
src_folder = f"{rootpath}/datasets/BingSearchResults"


def search_for_JSON(query_string: str) -> dict:
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


def generate_suggestion(query: str) -> str:
    json_obj = all_search_results.get(query)
    if not json_obj:
        return query

    web_pages = json_obj.get("webPages", {})
    results = web_pages.get("value", [])
    titles = [result["name"] for result in results]

    altered_query = json_obj.get("queryContext", {}).get("alteredQuery")
    if altered_query:
        best_match = get_best_title_match(altered_query, titles)
        if best_match is not None:
            return best_match
        return (
            html.unescape(get_best_title_match(query, titles))
            if titles
            else altered_query
        )

    return html.unescape(get_best_title_match(query, titles)) if titles else query


check_spellchecker(generate_suggestion, only_hard=False, case_sensitive=True)
