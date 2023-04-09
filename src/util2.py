from datetime import datetime
import json
import os
import pickle
import threading
import time
from typing import Literal, Union
from pathlib import Path
import csv
from rich import progress as prog

from preprocessing.suggester import generate_suggestion

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


def open_targets(dataset: Literal["test", "validation"], task: Literal["cea", "cpa", "cta"]):
    if dataset == "test":
        file_path = f"{ROOTPATH}/datasets/HardTablesR1/DataSets/HardTablesR1/Test/target/{task}_target.csv"
    elif dataset == "validation":
        file_path = f"{ROOTPATH}/datasets/HardTablesR1/DataSets/HardTablesR1/Valid/gt/{task}_gt.csv"
    else:
        raise ValueError(f"Invalid dataset: {dataset}")

    rows = get_csv_rows(file_path)

    targets = {}
    for row in rows:
        if not row[0] in targets:
            targets[row[0]] = set([int(row[2])])
        else:
            targets[row[0]].add(int(row[2]))

    return {key: list(value) for key, value in targets.items()}


def open_table(
    dataset: Literal["test", "validation"],
    file_name: str,
    entity_cols: list[int],
    spellcheck: Union[None, Literal["bing"]] = None,
):
    from classes2 import Cell, Column

    if file_name == "HXA71H9Q":
        x = 1

    if dataset == "test":
        file_path = f"{ROOTPATH}/datasets/HardTablesR1/DataSets/HardTablesR1/Test/tables"
    elif dataset == "validation":
        file_path = f"{ROOTPATH}/datasets/HardTablesR1/DataSets/HardTablesR1/Valid/tables"
    else:
        raise ValueError(f"Invalid dataset: {dataset}")

    rows = get_csv_rows(f"{file_path}/{file_name}.csv", skip_header=True)

    # check that every row has the same number of columns
    num_cols = len(rows[0])
    for row in rows:
        assert len(row) == num_cols

    # iterate over the columns and return the ones that are in cols
    columns: list[Column] = []
    literal_columns: list[list[str]] = []
    for i in range(num_cols):
        if i in entity_cols:
            if spellcheck == "bing":
                columns.append(Column([Cell(generate_suggestion(row[i])) for row in rows], i))
            else:
                columns.append(Column([Cell(row[i]) for row in rows]))
        else:
            literal_columns.append([row[i] for row in rows])

    return columns, literal_columns


def open_tables(
    dataset: Literal["test", "validation"],
    task: Literal["cea", "cpa", "cta"] = "cea",
    spellcheck: Union[None, Literal["bing"]] = None,
):
    from classes2 import Table, TableCollection

    if dataset == "test":
        raise NotImplementedError("Test dataset not implemented yet")
    elif dataset == "validation":
        gt_rows = get_csv_rows(f"{ROOTPATH}/datasets/HardTablesR1/DataSets/HardTablesR1/Valid/gt/{task}_gt.csv")

    targets = open_targets(dataset, task)
    tables = {}
    with progress:
        for filename, cols in progress.track(targets.items(), description=f"Opening {dataset} dataset"):
            columns, literal_columns = open_table(dataset, filename, cols, spellcheck=spellcheck)

            # set the correct id for each cell
            for i, col in zip(cols, columns):
                for j, cell in enumerate(col.cells):
                    gt_row = next(row for row in gt_rows if row[0] == filename and int(row[1]) == j + 1 and int(row[2]) == i)
                    cell.correct_id = int(gt_row[3].split("/")[-1][1:])

            tables[filename] = Table(columns, literal_columns)
    return TableCollection(tables)


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
        if not os.path.isfile(self.filename):
            with open(self.filename, "wb") as f:
                pickle.dump({}, f)

        with open(self.filename, "rb") as f:
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
        with open(self.filename, "wb") as f:
            pickle.dump(self.data, f)
    
    def close_data(self):
        self.save_data()
        self.data = None