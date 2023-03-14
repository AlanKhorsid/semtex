from util import pickle_load, pickle_save


import json
import os

result_dict = {}

folder_path = "/Users/alankhorsid/Documents/semtex/datasets/BingSearchResults"

for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        with open(os.path.join(folder_path, filename)) as json_file:
            json_data = json.load(json_file)
            original_query = json_data["queryContext"]["originalQuery"]
            result_dict[original_query] = json_data


# for key, value in result_dict.items():
#     print(f"{key}: {value}\n")


dict1 = result_dict
dict2 = pickle_load("all-test-cells-search-results", is_dump=True)

print(len({**dict1, **dict2}))
