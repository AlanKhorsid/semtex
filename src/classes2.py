from typing import Union
from _requests import wikidata_entity_search, wikidata_get_entity
from util2 import progress


class Candidate:
    id: int
    title: Union[str, None]
    description: Union[str, None]
    properties: Union[list[tuple[int, int]], None]

    def __init__(self, id: int):
        self.id = id

    def fetch_info(self):
        entity_data = wikidata_get_entity(self.id)
        self.title = entity_data["title"]
        self.description = entity_data["description"]
        self.properties = [(int(pair[0][1:]), int(pair[1][1:])) for pair in entity_data["properties"]]


class Cell:
    mention: str
    candidates: Union[list[Candidate], None]

    def __init__(self, mention: str):
        self.mention = mention
        self.candidates = None

    def fetch_candidates(self):
        if self.mention == "":
            self.candidates = []
            return

        entity_ids = wikidata_entity_search(self.mention)
        self.candidates = [Candidate(int(entity_id[1:])) for entity_id in entity_ids]


class Column:
    cells: list[Cell]

    def __init__(self, cells: list[Cell]):
        self.cells = cells


class Table:
    columns: list[Column]
    literal_columns: list[list[str]]

    def __init__(self, columns: list[Column], literal_columns: list[list[str]]):
        self.columns = columns
        self.literal_columns = literal_columns


class TableCollection:
    tables: dict[str, Table]

    def __init__(self, tables: dict[str, Table]):
        self.tables = tables

    def fetch_candidates(self):
        with progress:
            for table in progress.track(self.tables.values(), description="Fetching candidates"):
                for column in table.columns:
                    for cell in column.cells:
                        cell.fetch_candidates()

    def fetch_info(self):
        with progress:
            for table in progress.track(self.tables.values(), description="Fetching info"):
                for column in table.columns:
                    for cell in column.cells:
                        for candidate in cell.candidates:
                            candidate.fetch_info()
