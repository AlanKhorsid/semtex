from util import pickle_load, pickle_save


import json
import os

# result_dict = {}

# folder_path = "/Users/alankhorsid/Documents/semtex/datasets/BingSearchResults"

# for filename in os.listdir(folder_path):
#     if filename.endswith(".json"):
#         with open(os.path.join(folder_path, filename)) as json_file:
#             json_data = json.load(json_file)
#             original_query = json_data["queryContext"]["originalQuery"]
#             result_dict[original_query] = json_data


# # for key, value in result_dict.items():
# #     print(f"{key}: {value}\n")


# dict1 = result_dict
# dict2 = pickle_load("all-test-cells-search-results", is_dump=True)

# print(len({**dict1, **dict2}))


# def contains(substring, string):
#     return substring.lower() in string.lower()


# print(
#     contains(
#         "jacklyn",
#         "Jacklyn - Name Meaning, What does Jacklyn mean? - Think Baby".lower(),
#     )
# )

# from difflib import SequenceMatcher


# def similarity(a, b):
#     return SequenceMatcher(None, a, b).ratio()


# def best_suggestion(input_string, suggestions):
#     scores = [similarity(input_string, suggestion) for suggestion in suggestions]
#     best_match_index = scores.index(max(scores))
#     return suggestions[best_match_index]


# input_string = "j. addams"
# suggestions = ["John Adams", "Edward J. Adams", "Patrick J. Adams", "Wednesday Addams"]

# input_string = "j. j. vilsmayr"
# suggestions = [
#     "Johann Joseph Vilsmayr",
#     "Category:Films directed by Joseph Vilsmaier",
#     "Joseph Vilsmaier",
#     "Bergkristall",
#     "Irmgard Vilsmaier",
# ]

# input_string = "e. e. velkiers"
# suggestions = ["Esther Elizabeth Velkiers", "Emil Volkers", "Edward Cecil Villiers"]
# print(best_suggestion(input_string, suggestions))

x1 = pickle_load("25-03_10-27-34", is_dump=True)

print(x1["model"].get_params())
print(x1["f1"])
