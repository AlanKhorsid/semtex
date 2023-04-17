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
                norm_abs_diff = 1 - (abs(literal_as_number - statement_value_as_number)) / (
                    max(abs(literal_as_number), abs(statement_value_as_number))
                )
                return norm_abs_diff
            except:
                return 0
        elif self.type == "time":
            try:
                literal_date = parse_date(literal)
                if self.value.date() == literal_date.date():
                    return 1
                d1 = literal_date
                d2 = self.value
                if d2 < d1:
                    d1, d2 = d2, d1
                norm_date_diff = 1 - abs((d2 - d1).days) / ((date(d2.year, 12, 31) - date(d1.year, 1, 1)).days + 1)
                return norm_date_diff
            except:
                return 0

        elif self.type == "monolingualtext":
            if self.value == literal:
                return 1
            return Levenshtein.ratio(self.value, literal)
        else:
            return 0

    def get_entity_score(self, entity_mention: str):
        if self.type != "wikibase-item":
            return 0

        entity_data = get_entity(self.value)
        title = entity_data[0]
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
            elif type == "monolingualtext":
                value = value["text"]
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
    has_correct_candidate: Union[bool, None]

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
        self.has_correct_candidate = any(candidate.id == self.correct_id for candidate in self.candidates)

    def get_property_scores(self, row_entities: list["Cell"], row_literals: list[str]):
        if len(self.candidates) == 0:
            return [{"total": 0, "literal_scores": [], "entity_scores": [], "candidate": None}]

        scores = []
        for candidate in self.candidates:
            literal_scores = []
            for literal in row_literals:
                score, properties, _ = candidate.get_property_score(literal, type="literal")
                literal_scores.append((score, properties))
            entity_scores = []
            for col_index, entity in row_entities:
                score, properties, values = candidate.get_property_score(entity.mention, type="entity")
                entity_scores.append((score, properties, values, col_index))
            total = sum([score for score, _ in literal_scores]) + sum([score for score, _, _, _ in entity_scores])
            scores.append((total, literal_scores, entity_scores, candidate))
        scores.sort(key=lambda x: x[0], reverse=True)

        best_score = scores[0][0]
        best_scores = []

        for total, literal_scores, entity_scores, candidate in scores:
            if total == best_score:
                best_scores.append(
                    {
                        "total": total,
                        "literal_scores": [
                            {"score": score, "properties": properties} for score, properties in literal_scores
                        ],
                        "entity_scores": [
                            {"score": score, "properties": properties, "values": values, "col_index": col_index}
                            for score, properties, values, col_index in entity_scores
                        ],
                        "candidate": candidate,
                    }
                )
            else:
                break

        return best_scores


class Column:
    cells: list[Union[Cell, None]]
    index: int

    def __init__(self, cells: list[Union[Cell, None]], index: int):
        self.cells = cells
        self.index = index


class Table:
    columns: list[Column]
    literal_columns: list[list[Union[str, None]]]
    targets: dict

    def __init__(self, columns: list[Column], literal_columns: list[list[Union[str, None]]], targets: dict):
        self.columns = columns
        self.literal_columns = literal_columns
        self.targets = targets

    def dogboost(self):
        col_scores = []
        for i, column in enumerate(self.columns):
            best_scores = []
            for j, cell in enumerate(column.cells):
                row_literals = [literal_column[j] for literal_column in self.literal_columns]
                row_entities = [
                    (entity_column.index, entity_column.cells[j])
                    for z, entity_column in enumerate(self.columns)
                    if z != i
                ]
                best_score = cell.get_property_scores(row_entities, row_literals)
                best_scores.append(best_score)
            avg_score = sum([score[0]["total"] for score in best_scores]) / len(best_scores)
            col_scores.append({"avg_score": avg_score, "best_scores": best_scores, "column": column})
        col_scores.sort(key=lambda x: x["avg_score"], reverse=True)

        best_scores = col_scores[0]["best_scores"]
        column = col_scores[0]["column"]

        selections = []
        for i, c in enumerate(best_scores):
            best = None
            best_overlap = -1
            for cand in c:
                overlap_count = 0
                for j, other_c in enumerate(best_scores):
                    if i == j:
                        continue
                    for other_cand in other_c:
                        overlap_found = False
                        if other_cand["candidate"] is None:
                            continue
                        for z, literal in enumerate(cand["literal_scores"]):
                            for l in literal["properties"]:
                                if l in other_cand["literal_scores"][z]["properties"]:
                                    overlap_found = True
                        for z, entity in enumerate(cand["entity_scores"]):
                            for e in entity["properties"]:
                                if e in other_cand["entity_scores"][z]["properties"]:
                                    overlap_found = True
                        if overlap_found:
                            overlap_count += 1
                if overlap_count > best_overlap:
                    best = [cand]
                    best_overlap = overlap_count
                elif overlap_count == best_overlap:
                    best.append(cand)
            selections.append(best)

        for s in selections:
            if len(s) > 1:
                # If we have multiple candidates with same confidence, then we should give it ML
                # return 0, 0, 0
                pass

        cea_results = []
        num_correct_annotations = 0
        num_submitted_annotations = 0
        num_ground_truth_annotations = 0

        for i, (score, cell) in enumerate(zip(selections, column.cells)):
            num_ground_truth_annotations += 1
            if score[0]["candidate"] is None:
                for entity in score[0]["entity_scores"]:
                    num_ground_truth_annotations += 1
                continue

            cea_results.append((i + 1, column.index, score[0]["candidate"]))
            for entity in score[0]["entity_scores"]:
                if len(entity["values"]) > 1 or len(entity["values"]) > 1:
                    if len(set(entity["values"])) != 1:
                        # Here we should select the one with the most occouring property - not just the first one
                        pass

                num_ground_truth_annotations += 1
                num_submitted_annotations += 1

                entity_col = [c for c in self.columns if c.index == entity["col_index"]][0]
                c = [c for c in entity_col.cells[i].candidates if c.id == entity["values"][0]]
                if len(c) == 0:
                    # fetch candidate here
                    continue
                else:
                    c = c[0]

                cea_results.append((i + 1, entity["col_index"], c))

                if entity_col.cells[i].correct_id == entity["values"][0]:
                    num_correct_annotations += 1

            # if score[0]["total"] < (len(score[0]["literal_scores"]) + len(score[0]["entity_scores"])) * 0.8:
            #     x = 1
            num_submitted_annotations += 1
            if score[0]["candidate"].id == cell.correct_id:
                num_correct_annotations += 1
                # diff = (len(score[0]["literal_scores"]) + len(score[0]["entity_scores"])) - score[0]["total"]
                # if diff > 0.1:
                #     x = 1
            # else:
            #     if cell.has_correct_candidate and len(score) == 1:
            #         x = 1

            # If confidence (total) is less than some threshold, then we should give it ML
            # If title is not close to the cell mention, then we should give it ML

        return cea_results, num_correct_annotations, num_submitted_annotations, num_ground_truth_annotations


class TableCollection:
    tables: dict[str, Table]

    def __init__(self, tables: dict[str, Table]):
        self.tables = tables

    def fetch_candidates(self):
        with progress:
            for table in progress.track(self.tables.values(), description="Fetching candidates"):
                for column in table.columns:
                    for cell in column.cells:
                        if cell is not None:
                            cell.fetch_candidates()

    def fetch_info(self):
        candidate_ids = set()
        for table in self.tables.values():
            for column in table.columns:
                for cell in column.cells:
                    if cell is not None:
                        for candidate in cell.candidates:
                            candidate_ids.add(candidate.id)
        wikidata_fetch_entities(list(candidate_ids))

        for table in self.tables.values():
            for column in table.columns:
                for cell in column.cells:
                    if cell is not None:
                        for candidate in cell.candidates:
                            candidate.fetch_info()

    def fetch_statement_entities(self):
        entity_ids = set()
        for table in self.tables.values():
            for column in table.columns:
                for cell in column.cells:
                    if cell is not None:
                        for candidate in cell.candidates:
                            for statement in candidate.statements:
                                if statement.type == "wikibase-item":
                                    entity_ids.add(statement.value)
        wikidata_fetch_entities(list(entity_ids))

    def limit_to(self, n: int):
        x = list(self.tables.items())[:n]
        self.tables = dict(x)
