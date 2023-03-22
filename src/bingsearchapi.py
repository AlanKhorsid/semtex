import requests
from decouple import config
from util import pickle_load, pickle_save, pickle_save_in_folder

subscription_key = config("subscription_key_S2", default="")
search_url = "https://api.bing.microsoft.com/v7.0/search"
headers = {"Ocp-Apim-Subscription-Key": subscription_key}


# Define a function to get the webpage in JSON format
def get_webpage(query: str):
    search_query = query + "site:www.wikidata.org"
    params = {"q": search_query, "textDecorations": True, "textFormat": "HTML"}
    try:
        response = requests.get(search_url, headers=headers, params=params)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {e.response.text}")
        return None

    search_results = response.json()
    return search_results


def compute_bing_search_results():
    cells = pickle_load("all_test_cells", is_dump=True)
    print(f"Loaded {len(cells)} mentions from pickle file")
    search_results = {}
    for i, cell in enumerate(cells):
        if cell in search_results:
            print(f"Processed {i+1}/{len(cells)} mentions; already in pickle file")
            continue
        else:
            search_results[cell] = get_webpage(cell)
            pickle_save_in_folder(
                search_results[cell], "search_results_for_tests_cell_april"
            )
            print(f"Processed {i+1}/{len(cells)} mentions; saved to pickle file")
    print(f"Now proceeding to save all search results to one pickle file")
    pickle_save(search_results)
    print(f"Finished processing all mentions")
