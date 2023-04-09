from datetime import datetime, date
from typing import Literal, Union
from _requests import wikidata_entity_search, wikidata_get_entity, wikidata_fetch_entities, get_entity
from util2 import parse_entity_description, parse_entity_statements, parse_entity_title, progress
from dateutil.parser import parse as parse_date
import Levenshtein

class Statement:
    property: int
    type: str
    value: Union[str, None]

    def __init__(self, property: int, type: str, value: Union[str, None] = None):
        self.property = property
        self.type = type
        self.value = value
    
    def get_literal_score(self, literal: str):
        if self.type == "quantity":
            if self.value == literal:
                return 1
            try:
                literal_as_number = float(literal)
                statement_value_as_number = float(self.value)
                norm_abs_diff = 1 - (abs(literal_as_number - statement_value_as_number)) / (max(abs(literal_as_number), abs(statement_value_as_number)))
                return norm_abs_diff
            except:
                return 0
        elif self.type == "time":
            try:
                literal_date = parse_date(literal)
            except:
                return 0
            if self.value.date() == literal_date.date():
                return 1
            d1 = literal_date
            d2 = self.value
            if d2 < d1:
                d1, d2 = d2, d1
            norm_date_diff = 1 - abs((d2 - d1).days) / ((date(d2.year, 12, 31) - date(d1.year, 1, 1)).days + 1)
            return norm_date_diff
        else:
            return 0
    
    def get_entity_score(self, entity_mention: str):
        if self.type != "wikibase-item":
            return 0
        
        entity_data = get_entity(self.value)
        title = parse_entity_title(entity_data)
        if title is None:
            return 0
        if title == entity_mention:
            return 1
        return Levenshtein.ratio(title, entity_mention)
        

class Candidate:
    id: int
    title: Union[str, None]
    description: Union[str, None]
    statements = Union[list[Statement], None]

    def __init__(self, id: int):
        self.id = id

    def fetch_info(self):
        title, description, statements = get_entity(self.id)
        self.title = title
        self.description = description
        self.statements = self.parse_statements(statements)
        y = 1
    
    @staticmethod
    def parse_statements(statements: list):
        parsed_statements = []
        for prop, type, value in statements:
            if type == "wikibase-item":
                value = int(value["id"][1:])
            elif type == "quantity":
                value = value["amount"]
                if value[0] == "+":
                    value = value[1:]
            elif type == "time":
                time_str = value["time"]
                try:
                    dt = datetime.strptime(time_str, "+%Y-%m-%dT%H:%M:%SZ")
                except ValueError:
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
            parsed_statements.append(Statement(prop, type, value))
        return parsed_statements

    def get_property_score(self, value: str, type: Literal["literal", "entity"]):
        score = 0
        properties = []
        values = []

        for statement in self.statements:
            val_score = statement.get_literal_score(value) if type == "literal" else statement.get_entity_score(value)
            if val_score > score:
                score = val_score
                properties = [statement.property]
                values = [statement.value]
            elif val_score == score:
                properties.append(statement.property)
                values.append(statement.value)

        return score, properties, values


class Cell:
    mention: str
    correct_id: Union[int, None]
    candidates: Union[list[Candidate], None]

    def __init__(self, mention: str):
        self.mention = mention
        self.candidates = None
        self.correct_id = None

    def fetch_candidates(self):
        if self.mention == "":
            self.candidates = []
            return

        entity_ids = wikidata_entity_search(self.mention)
        self.candidates = [Candidate(int(entity_id[1:])) for entity_id in entity_ids]
    
    def get_property_scores(self, row_entities: list["Cell"], row_literals: list[str]):
        if len(self.candidates) == 0:
            return {
                "total": 0,
                "literal_scores": [],
                "entity_scores": [],
                "candidate": None
            }

        scores = []
        for candidate in self.candidates:
            literal_scores = []
            for literal in row_literals:
                score, properties, _ = candidate.get_property_score(literal, type="literal")
                literal_scores.append((score, properties))
            entity_scores = []
            for entity in row_entities:
                score, properties, values = candidate.get_property_score(entity.mention, type="entity")
                entity_scores.append((score, properties, values))
            total = sum([score for score, _ in literal_scores]) + sum([score for score, _, _ in entity_scores])
            scores.append((total, literal_scores, entity_scores, candidate))
        scores.sort(key=lambda x: x[0], reverse=True)

        total, literal_scores, entity_scores, candidate = scores[0]
        return {
            "total": total,
            "literal_scores": [{"score": score, "properties": properties} for score, properties in literal_scores],
            "entity_scores": [{"score": score, "properties": properties, "values": values} for score, properties, values in entity_scores],
            "candidate": candidate
        }

        # literal_scores = []
        # for candidate in self.candidates:
        #     candidate_scores = []
        #     for literal in row_literals:
        #         score, properties = candidate.get_property_score(literal, type="literal")
        #         candidate_scores.append((score, properties))
        #     literal_scores.append((candidate_scores, candidate))
        # literal_scores.sort(key=lambda x: sum([score for score, _ in x[0]]), reverse=True)
        
        # entity_scores = []
        # for candidate in self.candidates:
        #     candidate_scores = []
        #     for entity in row_entities:
        #         score, properties = candidate.get_property_score(entity.mention, type="entity")
        #         candidate_scores.append((score, properties))
        #     entity_scores.append((candidate_scores, candidate))
        # entity_scores.sort(key=lambda x: sum([score for score, _ in x[0]]), reverse=True)

        # # combine scores by candidate
        # combined_scores = []
        # for i, literal_score in enumerate(literal_scores):
        #     for j, entity_score in enumerate(entity_scores):
        #         if literal_score[1] == entity_score[1]:
        #             total = 0
        #             for n in literal_score[0]:
        #                 total += n[0]
        #             for n in entity_score[0]:
        #                 total += n[0]
        #             combined_scores.append((total, literal_score[0], entity_score[0], literal_score[1]))
        #             break
        
        # # sort by sum
        # combined_scores.sort(key=lambda x: x[0], reverse=True)

        
        


class Column:
    cells: list[Cell]
    index: int

    def __init__(self, cells: list[Cell], index: int):
        self.cells = cells
        self.index = index


class Table:
    columns: list[Column]
    literal_columns: list[list[str]]

    def __init__(self, columns: list[Column], literal_columns: list[list[str]]):
        self.columns = columns
        self.literal_columns = literal_columns
    
    def dogboost(self):
        col_scores = []
        for i, column in enumerate(self.columns):
            best_scores = []
            for j, cell in enumerate(column.cells):
                row_literals = [literal_column[j] for literal_column in self.literal_columns]
                row_entities = [entity_column.cells[j] for z, entity_column in enumerate(self.columns) if z != i]
                best_score = cell.get_property_scores(row_entities, row_literals)
                best_scores.append(best_score)
            avg_score = sum([score["total"] for score in best_scores]) / len(best_scores)
            col_scores.append({"avg_score": avg_score, "best_scores": best_scores, "column": column})
        col_scores.sort(key=lambda x: x["avg_score"], reverse=True)

        best_scores = col_scores[0]["best_scores"]
        column = col_scores[0]["column"]

        num_correct_annotations = 0
        num_submitted_annotations = 0
        num_ground_truth_annotations = 0

        for score, cell in zip(best_scores, column.cells):
            num_ground_truth_annotations += 1
            if score["candidate"] is None:
                continue
            
            num_submitted_annotations += 1

            if score["candidate"].id == cell.correct_id:
                num_correct_annotations += 1

            for literal_score in score["literal_scores"]:
                if literal_score["score"] < 0.9:
                    x = 1
            for entity_score in score["entity_scores"]:
                if entity_score["score"] < 0.9:
                    x = 1
        
        precision = num_correct_annotations / num_submitted_annotations if num_submitted_annotations > 0 else 0
        recall = num_correct_annotations / num_ground_truth_annotations if num_ground_truth_annotations > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

        x = 1


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
        candidate_ids = set()
        for table in self.tables.values():
            for column in table.columns:
                for cell in column.cells:
                    for candidate in cell.candidates:
                        candidate_ids.add(candidate.id)
        wikidata_fetch_entities(list(candidate_ids))

        for table in self.tables.values():
            for column in table.columns:
                for cell in column.cells:
                    for candidate in cell.candidates:
                        candidate.fetch_info()

    def fetch_statement_entities(self):
        entity_ids = set()
        for table in self.tables.values():
            for column in table.columns:
                for cell in column.cells:
                    for candidate in cell.candidates:
                        for statement in candidate.statements:
                            if statement.type == "wikibase-item":
                                entity_ids.add(statement.value)
        wikidata_fetch_entities(list(entity_ids))

    
    def limit_to(self, n: int):
        x = list(self.tables.items())[:n]
        self.tables = dict(x)
