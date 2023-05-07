from datetime import datetime
import json
import os
import pickle
import shutil
import string
import threading
import time
from typing import Dict, List, Literal, Union
from pathlib import Path
import csv
from rich import progress as prog
from nltk.corpus import stopwords

from preprocessing.suggester import generate_suggestion, release_search_results

ROOTPATH = Path(__file__).parent.parent

progress = prog.Progress(
    prog.SpinnerColumn(),
    prog.TextColumn("[progress.description]{task.description}"),
    prog.BarColumn(),
    prog.TaskProgressColumn(),
    prog.TextColumn("[red]{task.completed} of [red]{task.total} done"),
    prog.TextColumn("[yellow]Elapsed:"),
    prog.TimeElapsedColumn(),
    prog.TextColumn("[cyan]ETA:"),
    prog.TimeRemainingColumn(),
)


def get_csv_rows(file_path: str, skip_header: bool = False) -> list:
    with open(file_path, "r", encoding="utf8") as f:
        reader = csv.reader(f)
        if skip_header:
            next(reader)
        return list(reader)


def open_targets(dataset: Literal["test", "validation"], year: Literal["2022", "2023"]):
    if year == "2022":
        if dataset == "validation":
            cea_path = f"{ROOTPATH}/datasets/HardTablesR1/DataSets/HardTablesR1/Valid/gt/cea_gt.csv"
            cta_path = f"{ROOTPATH}/datasets/HardTablesR1/DataSets/HardTablesR1/Valid/gt/cta_gt.csv"
            cpa_path = f"{ROOTPATH}/datasets/HardTablesR1/DataSets/HardTablesR1/Valid/gt/cpa_gt.csv"
        elif dataset == "test":
            cea_path = f"{ROOTPATH}/datasets/HardTablesR1/DataSets/HardTablesR1/Test/target/cea_target.csv"
            cta_path = f"{ROOTPATH}/datasets/HardTablesR1/DataSets/HardTablesR1/Test/target/cta_target.csv"
            cpa_path = f"{ROOTPATH}/datasets/HardTablesR1/DataSets/HardTablesR1/Test/target/cpa_target.csv"
    elif year == "2023":
        if dataset == "validation":
            cea_path = f"{ROOTPATH}/datasets/2023/DataSets/Valid/targets/cea_targets.csv"
            cta_path = f"{ROOTPATH}/datasets/2023/DataSets/Valid/targets/cta_targets.csv"
            cpa_path = f"{ROOTPATH}/datasets/2023/DataSets/Valid/targets/cpa_targets.csv"
        elif dataset == "test":
            cea_path = f"{ROOTPATH}/datasets/2023/DataSets/Test/targets/cea_targets.csv"
            cta_path = f"{ROOTPATH}/datasets/2023/DataSets/Test/targets/cta_targets.csv"
            cpa_path = f"{ROOTPATH}/datasets/2023/DataSets/Test/targets/cpa_targets.csv"

    cea_rows = get_csv_rows(cea_path)
    cta_rows = get_csv_rows(cta_path)
    cpa_rows = get_csv_rows(cpa_path)

    files = set()
    for row in cea_rows:
        files.add(row[0])
    for row in cta_rows:
        files.add(row[0])
    for row in cpa_rows:
        files.add(row[0])

    targets = {}
    with progress:
        # for file in progress.track(list(files)[:100], description="Opening targets"):
        for file in progress.track(files, description="Opening targets"):
            targets[file] = {
                "cea": [(int(row[1]), int(row[2])) for row in cea_rows if row[0] == file],
                "cta": [int(row[1]) for row in cta_rows if row[0] == file],
                "cpa": [(int(row[1]), int(row[2])) for row in cpa_rows if row[0] == file],
            }

    return targets


def open_table(
    dataset: Literal["test", "validation"],
    file_name: str,
    cea_targets: list,
    year: Literal["2022", "2023"],
    spellcheck: Union[None, Literal["bing"]] = None,
):
    from classes2 import Cell, Column

    if year == "2022":
        if dataset == "validation":
            file_path = f"{ROOTPATH}/datasets/HardTablesR1/DataSets/HardTablesR1/Valid/tables"
        elif dataset == "test":
            file_path = f"{ROOTPATH}/datasets/HardTablesR1/DataSets/HardTablesR1/Test/tables"
    elif year == "2023":
        if dataset == "validation":
            file_path = f"{ROOTPATH}/datasets/2023/DataSets/Valid/tables"
        elif dataset == "test":
            file_path = f"{ROOTPATH}/datasets/2023/DataSets/Test/tables"

    rows = get_csv_rows(f"{file_path}/{file_name}.csv", skip_header=True)

    # check that every row has the same number of columns
    num_cols = len(rows[0])
    for row in rows:
        assert len(row) == num_cols

    columns: list[Column] = []
    literal_columns: List[Dict[str, Union[List[Union[str, None]], int]]] = []
    for i in range(num_cols):
        col_is_entity = any(target[1] == i for target in cea_targets)
        if not col_is_entity:
            literal_columns.append({"cells": [row[i] if row[i] != "" else None for row in rows], "index": i})
            x = 1
        else:
            if spellcheck == "bing":
                cells = [
                    Cell(generate_suggestion(row[i], dataset=dataset, year=year)) if row[i] != "" else None
                    for row in rows
                ]
            else:
                cells = [Cell(row[i]) if row[i] != "" else None for row in rows]

            # check that every cell is a cea target
            for j, cell in enumerate(cells):
                if cell is not None:
                    assert (j + 1, i) in cea_targets

            columns.append(Column(cells, i))

    return columns, literal_columns


def open_tables(
    dataset: Literal["test", "validation"],
    task: Literal["cea", "cpa", "cta"] = "cea",
    year: Literal["2022", "2023"] = "2023",
    spellcheck: Union[None, Literal["bing"]] = None,
):
    from classes2 import Table, TableCollection

    if year == "2022":
        if dataset == "test":
            gt_rows = get_csv_rows(f"{ROOTPATH}/datasets/HardTablesR1/DataSets/HardTablesR1/Test/target/cea_target.csv")
        elif dataset == "validation":
            gt_rows = get_csv_rows(f"{ROOTPATH}/datasets/HardTablesR1/DataSets/HardTablesR1/Valid/gt/cea_gt.csv")
    elif year == "2023":
        if dataset == "test":
            gt_rows = None
        elif dataset == "validation":
            gt_rows = get_csv_rows(f"{ROOTPATH}/datasets/2023/DataSets/Valid/gt/cea_gt.csv")
    else:
        raise ValueError(f"Invalid year: {year}")

    targets = open_targets(dataset, year)
    tables = {}
    with progress:
        for filename, targets in progress.track(targets.items(), description=f"Opening {dataset} dataset"):
            columns, literal_columns = open_table(dataset, filename, targets["cea"], year, spellcheck=spellcheck)

            # set the correct id for each cell
            if gt_rows is not None:
                for col in columns:
                    for j, cell in enumerate(col.cells):
                        if cell is None:
                            continue
                        gt_row = next(
                            row
                            for row in gt_rows
                            if row[0] == filename and int(row[1]) == j + 1 and int(row[2]) == col.index
                        )
                        cell.correct_id = int(gt_row[3].split("/")[-1][1:])

            tables[filename] = Table(columns, literal_columns, targets)
    release_search_results()
    return TableCollection(tables)


def remove_stopwords(unfiltered_string: str) -> str:
    translator = str.maketrans("", "", string.punctuation)
    filtered_words = unfiltered_string.translate(translator)
    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in filtered_words.split() if word.lower() not in stop_words]
    return " ".join(filtered_words)


# STUSUFSUFSUF
# ----------------------------
# ----------------------------
# ----------------------------


def pickle_save(obj, filename: Union[str, None] = None):
    if os.path.isdir(f"{ROOTPATH}/src/pickle-dumps") == False:
        os.mkdir(f"{ROOTPATH}/src/pickle-dumps")

    if filename is not None:
        filename = f"{ROOTPATH}/src/pickle-dumps/{filename}.pickle"
    else:
        now = datetime.now()
        filename = f"{ROOTPATH}/src/pickle-dumps/{now.strftime('%d-%m_%H-%M-%S')}.pickle"
        i = 1
        while os.path.isfile(filename):
            filename = f"{ROOTPATH}/src/pickle-dumps/{now.strftime('%d-%m_%H-%M-%S')}_{i}.pickle"
            i += 1

    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def pickle_load(filename, is_dump: bool = False):
    file = f"{ROOTPATH}/src/{'pickle-dumps' if is_dump else 'pickles'}/{filename}.pickle"
    with open(file, "rb") as f:
        return pickle.load(f)


def parse_entity_title(entity_data: dict) -> Union[str, None]:
    try:
        return entity_data["labels"]["en"]["value"]
    except KeyError:
        return None


def parse_entity_description(entity_data: dict) -> Union[str, None]:
    try:
        return entity_data["descriptions"]["en"]["value"]
    except KeyError:
        return None


def parse_entity_statements(entity_data: dict):
    from classes2 import Statement

    assert "claims" in entity_data, "Entity data does not contain claims"

    statements: list[Statement] = []
    for claims in entity_data["claims"].values():
        for claim in claims:
            if claim["mainsnak"]["snaktype"] == "novalue" or claim["mainsnak"]["snaktype"] == "somevalue":
                continue

            type = claim["mainsnak"]["datatype"]
            if type == "wikibase-item":
                value = int(claim["mainsnak"]["datavalue"]["value"]["id"][1:])
            elif type == "quantity":
                value = claim["mainsnak"]["datavalue"]["value"]["amount"]
                if value[0] == "+":
                    value = value[1:]
                assert value[0] == "-" or value[0].isdigit(), f"Invalid value: {value}"
            elif type == "time":
                time_str = claim["mainsnak"]["datavalue"]["value"]["time"]
                try:
                    dt = datetime.strptime(time_str, "+%Y-%m-%dT%H:%M:%SZ")
                except ValueError:
                    # assert time_str[6:8] == "00" or time_str[9:11] == "00", f"Weird stuff here man: {time_str}"
                    if time_str[6:8] == "00":
                        time_str = time_str[:6] + "01" + time_str[8:]
                    if time_str[9:11] == "00":
                        time_str = time_str[:9] + "01" + time_str[11:]
                    try:
                        dt = datetime.strptime(time_str, "+%Y-%m-%dT%H:%M:%SZ")
                    except ValueError:
                        # some weird dates can occour. Should be fixed in the future
                        continue
                value = dt
            else:
                continue

            property = int(claim["mainsnak"]["property"][1:])
            statements.append(Statement(property, type, value))

    return statements


def parse_date(date_string: str) -> datetime:
    if date_string[0] == "+":
        date_string = date_string[1:]
        date = datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%SZ")
    elif date_string[0] == "-":
        date_string = date_string[1:]
        date = datetime.strptime(date_string, "%Y%m%d-%H:%M:%SZ")
    else:
        raise ValueError("Invalid date string: " + date_string)
    return date


def parse_entity_properties(entity_data: dict) -> dict:
    properties = []
    for claims in entity_data["claims"].values():
        for claim in claims:
            try:
                if (
                    claim["mainsnak"]["snaktype"] == "novalue"
                    or claim["mainsnak"]["snaktype"] == "somevalue"
                    or claim["mainsnak"]["datatype"] != "wikibase-item"
                ):
                    continue

                prop = claim["mainsnak"]["property"]
                target = claim["mainsnak"]["datavalue"]["value"]["id"]
                properties.append((prop, target))
            except KeyError:
                continue
            except:
                print("----- ERROR -------")
                print(claims)

    return properties


model = None

def predict_candidates(candidates):
    global model
    if model is None:
        model_info = pickle_load("best-model-so-far", is_dump=True)
        model = model_info["model"]
    
    index_of_best_candidate = None
    best_score = -1
    for i, candidate in enumerate(candidates):
        if candidate is None:
            continue
        pred = model.predict_proba(candidate.features)[1]
        if pred > best_score:
            best_score = pred
            index_of_best_candidate = i
    return index_of_best_candidate


class JsonUpdater:
    def __init__(self, filename):
        self.filename = f"{ROOTPATH}{filename}"
        self.lock = threading.Lock()
        self.data = None
        self.last_save_time = time.time()

    @property
    def data_loaded(self):
        return self.data is not None

    def load_data(self):
        with open(self.filename, "r") as f:
            self.data = json.load(f)

    def update_data(self, key, value):
        with self.lock:
            self.data[key] = value
            current_time = time.time()
            if current_time - self.last_save_time > 0:
                self.save_data()
                self.last_save_time = current_time

    def delete_data(self, key):
        with self.lock:
            del self.data[key]
            self.save_data()

    def save_data(self):
        with open(self.filename, "w") as f:
            json.dump(self.data, f)


class PickleUpdater:
    def __init__(self, filename, rootpath=ROOTPATH, save_interval=60):
        self.filename = f"{rootpath}{filename}"
        self.lock = threading.Lock()
        self.data = None
        self.last_save_time = time.time()
        self.save_interval = save_interval

    @property
    def data_loaded(self):
        return self.data is not None

    def load_data(self):
        if not os.path.isfile(f"{self.filename}.pickle"):
            with open(f"{self.filename}.pickle", "wb") as f:
                pickle.dump({}, f)

        with open(f"{self.filename}.pickle", "rb") as f:
            self.data = pickle.load(f)

    def update_data(self, key, value, force_save=False):
        with self.lock:
            self.data[key] = value
            current_time = time.time()
            if force_save or current_time - self.last_save_time > self.save_interval:
                self.save_data()
                self.last_save_time = current_time

    def delete_data(self, key):
        with self.lock:
            del self.data[key]
            self.save_data()

    def save_data(self):
        shutil.copyfile(f"{self.filename}.pickle", f"{self.filename}_backup.pickle")

        with open(f"{self.filename}.pickle", "wb") as f:
            pickle.dump(self.data, f)

    def close_data(self):
        self.save_data()
        self.data = None
