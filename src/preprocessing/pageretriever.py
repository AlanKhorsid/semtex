import csv
import json
import os
import requests
from decouple import config

# subscription_key = config("subscription_key", default="")
# search_url = "https://api.bing.microsoft.com/v7.0/search"
# headers = {"Ocp-Apim-Subscription-Key": subscription_key}


# Define a function to get the webpage in JSON format
# def get_webpage(query):
#     params = {"q": query, "textDecorations": True, "textFormat": "HTML"}
#     try:
#         response = requests.get(search_url, headers=headers, params=params)
#         response.raise_for_status()
#     except requests.exceptions.HTTPError as e:
#         print(f"HTTP error occurred: {e.response.text}")
#         return None

#     search_results = response.json()
#     return search_results


# specify the path to the directory containing the csv files
csv_dir = "/Users/alankhorsid/Documents/semtex/datasets/HardTablesR1/DataSets/HardTablesR1/Test/tables"

# specify the path to the cea_gt file
cea_gt_file = "/Users/alankhorsid/Documents/semtex/datasets/HardTablesR1/DataSets/HardTablesR1/Test/gt/cea_gt.csv"

# read the cea_gt file and store the information in a dictionary
cea_gt_info = {}
with open(cea_gt_file, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        filename = row[0]
        i = int(row[1])
        j = int(row[2])
        cell = row[3]
        # replace the double quotes within the cell value with escaped quotes
        cell = cell.replace('"', '\\"')
        cea_gt_info[(filename, i, j)] = cell

# loop through the keys in the cea_gt_info dictionary
for key in cea_gt_info.keys():
    filename = key[0]
    i = key[1]
    j = key[2]
    cell = cea_gt_info[key]

    # construct the path to the csv file
    csv_path = os.path.join(csv_dir, f"{filename}.csv")

    # open the csv file and retrieve the cell at (i,j)
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
        cell_value = rows[i][j]

    # Append every cell value in one big txt file with newline
    with open("all_cells.csv", "a") as f:
        f.write(cell_value + "\n")
        print(f"{cell_value} written to file")
