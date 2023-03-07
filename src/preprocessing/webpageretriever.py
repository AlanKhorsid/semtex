import os
import json
import requests
from decouple import config

subscription_key = config("subscription_key", default="")
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


# Navigate to the directory containing the text files
text_files_dir = "src/preprocessing/tables"

# Create a directory for the output files
os.makedirs("webpages-json", exist_ok=True)

# Iterate over the text files
for filename in os.listdir(text_files_dir):
    # Open the text file and for each field in the text file get the webpage (skip the first line)
    with open(os.path.join(text_files_dir, filename), "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            # for each field in the line/record  get the webpage
            fields = line.strip().split(",")
            webpages = []
            for field in fields:
                webpage = get_webpage(field)
                if webpage is not None:
                    webpages.append(webpage)
            # Combine the search responses into a single string and split by delimiter
            data = json.dumps(webpages)
            delimiter = '{"_type": "SearchResponse"'
            split_data = data.split(delimiter)
            # Create a separate JSON file for each search response
            for j, search_response_str in enumerate(split_data):
                # Skip any split data that doesn't contain the search response type
                if not search_response_str.startswith(delimiter):
                    continue
                # Convert the search response string to JSON and write to file
                search_response = json.loads(delimiter + search_response_str)
                output_filename = f"{os.path.splitext(filename)[0]}_{i}_{j}.json"
                output_filepath = os.path.join("output", output_filename)
                with open(output_filepath, "w") as g:
                    json.dump(search_response, g)
