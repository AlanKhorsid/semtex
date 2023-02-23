import csv
import requests


def get_csv_lines(filename: str) -> list[list[str]]:
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        return list(reader)


def get_id_from_url(url: str) -> str:
    return url.split("/")[-1]


def fetch_entity_label(entity_id: str, lang: str = "en") -> str:
    API_URL = "https://www.wikidata.org/wiki/Special:EntityData"

    data = requests.get(f"{API_URL}/{entity_id}")
    english_label = data.json()["entities"][entity_id]["labels"][lang]["value"]
    return english_label


def append_to_csv(filename: str, line: str):
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"{line}\n")


def generate_vals_csv():
    cea_gt = get_csv_lines("./datasets/HardTablesR1/DataSets/HardTablesR1/Valid/gt/cea_gt.csv")

    current_file_lines = []
    current_filename = ""

    for line in cea_gt:
        filename, row, col, url = line

        # this entity's id has been renamed
        if url == "http://www.wikidata.org/entity/Q85786531":
            continue

        if current_filename != filename:
            current_filename = filename
            current_file_lines = get_csv_lines(
                f"./datasets/HardTablesR1/DataSets/HardTablesR1/Valid/tables/{filename}.csv"
            )

        id = get_id_from_url(url)
        actual_label = fetch_entity_label(id)
        label = current_file_lines[int(row)][int(col)]

        append_to_csv("./datasets/spellCheck/vals.csv", f'"{label}","{actual_label}"')

        percent_done = (cea_gt.index(line) / len(cea_gt)) * 100
        print(f"{percent_done:.2f}%   {label} -> {actual_label}")


def check_spellchecker(func, case_sensitive: bool = False, only_hard: bool = False):
    vals = get_csv_lines("./datasets/spellCheck/vals.csv")

    if only_hard:
        vals = [line for line in vals if line[0] != line[1]]

    total = len(vals)
    correct = 0

    for line in vals:
        label, actual_label = line
        if not case_sensitive:
            label = label.lower()
            actual_label = actual_label.lower()

        try:
            prediction = func(label)
        except:
            print(f"ERROR: {label} -> {actual_label}")
            continue

        if not case_sensitive:
            prediction = prediction.lower()

        if prediction == actual_label:
            correct += 1
            print(f"CORRECT: {label} -> {actual_label}")
        else:
            print(f"INCORRECT: {label} -> {prediction} (should be {actual_label})")

    print(f"Accuracy: {correct / total * 100:.2f}%")
