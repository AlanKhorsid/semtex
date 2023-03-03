import requests
import os
import json
from decouple import config
from bestmatch import get_best_title_match
from preprocesschecker import check_spellchecker_threaded

#subscription_key = config("subscription_key", default="")
#search_url = "https://api.bing.microsoft.com/v7.0/search"
#headers = {"Ocp-Apim-Subscription-Key": subscription_key}


def generate_suggestion(query="", use_api=False, filepath=""):
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
    

# check_spellchecker_threaded(generate_suggestion, num_threads=100)
