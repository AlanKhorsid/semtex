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
                columns.append(Column([generate_suggestion(row[i]) for row in rows]))
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

    targets = open_targets(dataset, task)
    tables = {}
    with progress:
        for filename, cols in progress.track(targets.items(), description=f"Opening {dataset} dataset"):
            columns, literal_columns = open_table(dataset, filename, cols, spellcheck=spellcheck)
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
    """
    Parses the title of an entity from the Wikidata API. If the entity has no
    English label, returns None.

    Parameters
    ----------
    entity_data : dict
        The entity data to parse.

    Returns
    -------
    str
        The title of the entity.

    Example
    -------
    >>> parse_entity_title({"labels": {"en": {"value": "Barack Obama"}}})
    "Barack Obama"
    """

    try:
        return entity_data["labels"]["en"]["value"]
    except KeyError:
        return None


def parse_entity_description(entity_data: dict) -> Union[str, None]:
    """
    Parses the description of an entity from the Wikidata API. If the entity has
    no English description, returns None.

    Parameters
    ----------
    entity_data : dict
        The entity data to parse.

    Returns
    -------
    str
        The description of the entity.

    Example
    -------
    >>> parse_entity_description({"descriptions": {"en": {"value": "44th President of the United States"}}})
    "44th President of the United States"
    """

    try:
        return entity_data["descriptions"]["en"]["value"]
    except KeyError:
        return None


def parse_entity_properties(entity_data: dict) -> dict:
    """
    Parses the claims of an entity from the Wikidata API.

    Parameters
    ----------
    entity_data : dict
        The entity data to parse.

    Returns
    -------
    dict
        The claims of the entity.

    Example
    -------
    >>> parse_entity_claims(entity_data)
    [
        ("P31", "Q5"),
        ("P21", "Q6581072"),
    ]
    """

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
        self.data = self.load_data()
        self.last_save_time = time.time()

    def load_data(self):
        with open(self.filename, "r") as f:
            return json.load(f)

    def update_data(self, key, value):
        with self.lock:
            self.data[key] = value
            current_time = time.time()
            if current_time - self.last_save_time > 20:
                self.save_data()
                self.last_save_time = current_time

    def delete_data(self, key):
        with self.lock:
            del self.data[key]
            self.save_data()

    def save_data(self):
        with open(self.filename, "w") as f:
            json.dump(self.data, f)
