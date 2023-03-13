import requests
from decouple import config
from util import pickle_load, pickle_save, pickle_save_in_folder

subscription_key = config("subscription_key_S2", default="")
search_url = "https://api.bing.microsoft.com/v7.0/search"
headers = {"Ocp-Apim-Subscription-Key": subscription_key}


# Define a function to get the webpage in JSON format
def get_webpage(query):
    params = {"q": query, "textDecorations": True, "textFormat": "HTML"}
    try:
        response = requests.get(search_url, headers=headers, params=params)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {e.response.text}")
        return None

    search_results = response.json()
    return search_results


######### Using Pickle to load test data #########
def pickle_search_results():
    cells = pickle_load("all_test_cells")
    all_searched_results = pickle_load("13-03_00-37-17")
    print(f"Loaded {len(cells)} mentions from pickle file")
    search_results = {}
    for i, cell in enumerate(cells[20000:]):  # limit to first 20000 cells
        if cell not in search_results:
            search_results[cell] = get_webpage(cell)
            pickle_save_in_folder(search_results[cell], "all_test_cells_search_results")
            print(f"Processed {i+1}/{len(cells)} mentions; saved to pickle file")
        else:
            print(f"Processed {i+1}/{len(cells)} mentions; already in pickle file")
    print(f"Now proceeding to save all search results to one pickle file")
    pickle_save(search_results)
    print(f"Finished processing all mentions")


# pickle_search_results()


def print_pickle_results():
    search_results = pickle_load("13-03_00-37-17")
    print(f"Loaded {len(search_results)} mentions from pickle file")
    for i, (cell, results) in enumerate(search_results.items()):
        if results is not None:
            print(f"Cell: {cell}")
            print(f"results: {results}")
            print(f"-------------------")
        else:
            print(f"Cell: {cell}")
            print(f"Webpage: None")
            print(f"Snippet: None")
            print(f"Rank: None")
            print(f"-------------------")


print_pickle_results()
