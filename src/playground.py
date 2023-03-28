import pprint
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

# x1 = pickle_load("25-03_10-27-34", is_dump=True)

# print(x1["model"].get_params())
# print(x1["f1"])

import torch
from transformers import AutoTokenizer, AutoModel

# Load the pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Define an input name
input_name = "Barack Obama"

# Tokenize the input name
inputs = tokenizer(input_name, return_tensors="pt", padding=True, truncation=True)

# Get the embeddings for the input name
outputs = model(**inputs)
embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()

print(embedding)

# Output:
# [[-5.71222067e-01  8.45796764e-02 -5.52923083e-01  3.20193231e-01
#   -4.90824580e-02  5.30520439e-01  4.68297422e-01  7.84230828e-01
#   -7.37598658e-01 -1.98808551e-01  7.66557902e-02 -1.41676858e-01
#   -2.94779181e-01  5.69287896e-01  1.08569466e-01 -7.07368404e-02
#   -9.84662354e-01  4.83514160e-01 -8.86642933e-02 -1.81324244e-01

# The output represents the embedding for the input name "Barack Obama". Each element in the array is a number that represents a specific aspect of the input name's meaning according to the pre-trained BERT model.
# This embedding vector can be used to calculate semantic similarity between different input names using cosine similarity. The closer the cosine similarity value is to 1, the more similar the two input names are in terms of meaning.
# I hope this example helps to clarify how embeddings work!
