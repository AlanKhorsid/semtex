import re
import json
import requests
import os
import json
from decouple import config
from bestmatch import get_best_title_match
from preprocesschecker import check_spellchecker_threaded


def preprocess_query(query):
    # Define the regex pattern for invalid characters in file names
    pattern = r'[\\/*?:"<>|]'

    # Replace invalid characters with an underscore
    preprocessed_query = re.sub(pattern, "_", query)
    # lowercase the filename
    preprocessed_query = preprocessed_query.lower()
    return preprocessed_query

def generate_suggestion(query="", use_api=False, filepath=""):
    """
    Preprocesses a search query by checking for spelling errors and suggesting alternatives.

    Args:
        query (str): The original search query.
        use_api (bool): A flag indicating whether to use the Bing search API or read from a JSON file in the webpages-json folder.
                        Defaults to False.

    Returns:
        str: The suggested alternative search query, or the best match for the original query based on search result titles.

    """
    # check if query has invalid characters
    query_file = preprocess_query(query)
    try:
        with open(
            f"datasets/BingSearchFiles/{query_file}.json",
            "r",
        ) as f:
            json_obj = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {query_file}")
    try:
        suggestion = json_obj["queryContext"]["alteredQuery"]
        # check if suggestion is contained in any of the titles
        results = json_obj["webPages"]["value"]
        titles = [result["name"] for result in results]
        for title in titles:
            if query in title:
                return suggestion
        return get_best_title_match(query, titles)

    except:
        results = json_obj["webPages"]["value"]
        titles = [result["name"] for result in results]
        suggestion = get_best_title_match(query, titles)
        print(f"Found best match: {suggestion}")
        return suggestion

def generate_suggestion2(query="", use_api=False, filepath=""):
    """
    Preprocesses a search query by checking for spelling errors and suggesting alternatives.

    Args:
        query (str): Optional search query for api call.
        use_api (bool): Optional flag to determine if the bing api should be used.
        filepath (str): Optional filepath string for json cache location.

    Returns:
        str: The suggested alternative search query, or the best match for the original query based on search result titles.

    """
    #Use the api in case the api flag is set to true
    search_results = ""
    if use_api and query != "":
        params = {"q": query, "textDecorations": True, "textFormat": "HTML"}
        try:
            response = requests.get(search_url, headers=headers, params=params)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            # print(f"HTTP error occurred: {e}")
            return None

        search_results = response.json()

    #Use json cache if the api flag is set to false
    elif filepath != "":
        with open(os.path.abspath(filepath), "r") as jsonfile:
            search_results = json.load(jsonfile)

    #Error, no path is given and api is false
    else:
        print("========ERROR========")
        print("API flag is false or no query given and no filepath given!")

    try:
        suggestion = search_results["queryContext"]["alteredQuery"]
        print(f"Found best match: {suggestion}")
        return suggestion
    except:
        results = search_results["webPages"]["value"]
        titles = [result["name"] for result in results]
        suggestion = get_best_title_match(query, titles)
        print(f"Found best match: {suggestion}")
        return suggestion
    
subscription_key = config("subscription_key", default="")
search_url = "https://api.bing.microsoft.com/v7.0/search"
headers = {"Ocp-Apim-Subscription-Key": subscription_key}

check_spellchecker_threaded(generate_suggestion, num_threads=100, only_hard=True)

# generate_suggestion("john adolph flemer")
# generate_suggestion("yttrium-103")
