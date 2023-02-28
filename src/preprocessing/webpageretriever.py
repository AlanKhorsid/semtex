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
        return None

    search_results = response.json()
    return search_results


# Navigate to the directory containing the text files
text_files_dir = "src/preprocessing/tables"

# Create a directory for the output files
os.makedirs("output", exist_ok=True)

# Iterate over the text files
for filename in os.listdir(text_files_dir):
    # Open the text file and for each field in the text file get the webpage (skip the first line)
    with open(os.path.join(text_files_dir, filename), "r") as f:
        with open(os.path.join("output", filename), "w") as g:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                # for each field in the line/record  get the webpage
                fields = line.strip().split(",")
                webpages = []
                for field in fields:
                    webpage = get_webpage(field)
                    if webpage is not None:
                        webpages.append(json.dumps(webpage))
                # write the webpages to the output file
                output_line = ",".join(webpages)
                g.write(output_line)
                g.write("\n")
