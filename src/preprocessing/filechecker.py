from pathlib import Path
import os

def check_if_file_exists(filename: str) -> bool:
    """Takes an input filename and checks if it is in the datasets/webpages-json folder, returns boolean True or False."""
    filepath = str(Path(__file__).parent.parent.parent) + "/datasets/webpages-json/" + str(filename)
    return Path(filepath).is_file()

def check_missing_files() -> [str]:
    """Checks for all files in src/preprocessing/tables and checks if a file exists for each entity value, returns a list of missing entity value json files."""

    text_files_dir = str(Path(__file__).parent.parent.parent) +  "/src/preprocessing/tables"
    missing_files = []

    #Generates the filenames for each entity value json file to check if it exists
    for filename in os.listdir(text_files_dir):
        with open(os.path.join(text_files_dir, filename), "r") as f:
            stripped_filename = filename.strip(".csv")
            for i, line in enumerate(f):
                if i == 0:
                    continue
                #generates the name of a .json file to lookup
                json_filename = stripped_filename + "_" + str(i) + ".json"
                fields = line.strip().split(",")

                for field in fields:
                    if not check_if_file_exists(json_filename):
                        missing_files.append(json_filename)
                        #call suggester from here
                        print(json_filename)
    
    #returns the list of missing json files
    return missing_files



#print(check_if_file_exists("QIYBVSKQ_18.json"))
#print(check_if_file_exists("blerp"))
#print(len(check_missing_files()))
