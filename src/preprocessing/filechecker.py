from pathlib import Path
import os

def check_if_file_exists(filename: str) -> bool:
    filepath = str(Path(__file__).parent.parent.parent) + "/datasets/webpages-json/" + str(filename)
    return Path(filepath).is_file()

def read_existing_file(filename: str) -> str:
    pass

def check_missing_files() -> [str]:
    text_files_dir = str(Path(__file__).parent.parent.parent) +  "/src/preprocessing/tables"
    missing_files = []
    for filename in os.listdir(text_files_dir):
        with open(os.path.join(text_files_dir, filename), "r") as f:
            stripped_filename = filename.strip(".csv")
            for i, line in enumerate(f):
                if i == 0:
                    continue
                json_filename = stripped_filename + "_" + str(i) + ".json"
                fields = line.strip().split(",")

                for field in fields:
                    if not check_if_file_exists(json_filename):
                        missing_files.append(json_filename)
                        #call suggester from here
                        print(json_filename)

    return missing_files



#print(check_if_file_exists("QIYBVSKQ_18.json"))
#print(check_if_file_exists("blerp"))
#print(len(check_missing_files()))
