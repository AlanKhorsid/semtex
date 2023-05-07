from datetime import datetime, date
from typing import Dict, List, Literal, Union
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
                if literal_as_number == 0 and statement_value_as_number == 0:
                    return 1
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

    def get_property_scores(self, row_entities: list[Union["Cell", None]], row_literals: list[Union[str, None]]):
        if len(self.candidates) == 0:
            return [{"score": 0, "candidate": None, "literal_scores": [], "entity_scores": []}]

        candidate_scores = []
        for candidate in self.candidates:
            literal_scores = []
            total = 0
            for col_i, literal in row_literals:
                if literal is None:
                    literal_scores.append(None)
                    continue
                score, properties, _ = candidate.get_property_score(literal, type="literal")
                literal_scores.append((col_i, score, properties))
                total += score
            entity_scores = []
            for col_i, entity in row_entities:
                if entity is None:
                    entity_scores.append(None)
                    continue
                score, properties, values = candidate.get_property_score(entity.mention, type="entity")
                entity_scores.append((col_i, score, properties, values))
                total += score
            candidate_scores.append((total, literal_scores, entity_scores, candidate))
        candidate_scores.sort(key=lambda x: x[0], reverse=True)

        best_score = candidate_scores[0][0]
        best_candidates = []

        for total, literal_scores, entity_scores, candidate in candidate_scores:
            if total == best_score:
                best_candidates.append(
                    {
                        "score": total,
                        "candidate": candidate,
                        "literal_scores": [
                            None
                            if literal_score is None
                            else {"score": literal_score[1], "properties": literal_score[2], "index": literal_score[0]}
                            for literal_score in literal_scores
                        ],
                        "entity_scores": [
                            None
                            if entity_score is None
                            else {
                                "score": entity_score[1],
                                "properties": entity_score[2],
                                "values": entity_score[3],
                                "index": entity_score[0],
                            }
                            for entity_score in entity_scores
                        ],
                    }
                )
            else:
                break

        return best_candidates


class Column:
    cells: list[Union[Cell, None]]
    index: int

    def __init__(self, cells: list[Union[Cell, None]], index: int):
        self.cells = cells
        self.index = index


class Table:
    columns: list[Column]
    literal_columns: List[Dict[str, Union[List[Union[str, None]], int]]]
    targets: dict

    def __init__(
        self, columns: list[Column], literal_columns: List[Dict[str, Union[List[Union[str, None]], int]]], targets: dict
    ):
        self.columns = columns
        self.literal_columns = literal_columns
        self.targets = targets

    def dogboost(self):
        SUBJECT_COL_INDEX = 0

        # Score candidates for each cell and select the best one(s)
        best_candidate_scores = []
        subject_col = self.columns[SUBJECT_COL_INDEX]
        for j, cell in enumerate(subject_col.cells):
            row_entities = []
            for cea_target in self.targets["cea"]:
                row_i, col_i = cea_target
                if row_i == j + 1 and col_i != SUBJECT_COL_INDEX:
                    col = next((c for c in self.columns if c.index == col_i), None)
                    assert col is not None
                    assert col.cells[j] is not None
                    row_entities.append((col_i, col.cells[j]))
            row_literals = [
                (literal_column["index"], literal_column["cells"][j]) for literal_column in self.literal_columns
            ]
            best_candidate_score = cell.get_property_scores(row_entities, row_literals)
            best_candidate_scores.append(best_candidate_score)

        # get properties and their counts
        props = {}
        for candidate_scores in best_candidate_scores:
            unique_props = set()
            for score in candidate_scores:
                for entity_score in score["entity_scores"]:
                    if entity_score is not None and entity_score["score"] != 0:
                        unique_props.update(entity_score["properties"])
                for literal_score in score["literal_scores"]:
                    if literal_score is not None and literal_score["score"] != 0:
                        unique_props.update(literal_score["properties"])

            for p in unique_props:
                if p not in props:
                    props[p] = 1
                else:
                    props[p] += 1
        prop_list = sorted(props.items(), key=lambda x: x[1], reverse=True)

        def get_cpa_scores(score):
            cpa_scores = {}
            for score_type in ("entity_scores", "literal_scores"):
                for score_item in score[score_type]:
                    if score_item is not None:
                        cpa_scores[score_item["index"]] = {
                            "score": score_item["score"],
                            "properties": score_item["properties"],
                            "objects": score_item["values"] if score_type == "entity_scores" else None,
                        }
            return cpa_scores

        # get best candidate for each cell according to the properties
        chosen_candidates = []
        for candidate_scores in best_candidate_scores:
            # If there is only one candidate with the highest score, choose it
            if len(candidate_scores) == 1:
                chosen_candidates.append(
                    {
                        "score": candidate_scores[0]["score"],
                        "best_possible_score": len(candidate_scores[0]["entity_scores"])
                        + len(candidate_scores[0]["literal_scores"]),
                        "candidate": candidate_scores[0]["candidate"],
                        "cpa_scores": get_cpa_scores(candidate_scores[0]),
                        "reason": "best candidate" if candidate_scores[0]["candidate"] is not None else "no candidates",
                    }
                )
                continue

            # If there are multiple candidates with the highest score,
            # choose the one with the most occurrences of the most common property
            best_candidates = []
            p_found = False
            prev_cnt = None
            for p, cnt in prop_list:
                if p_found and cnt != prev_cnt:
                    break
                for scores in candidate_scores:
                    p_found_in_candidate = False
                    for entity_score in scores["entity_scores"]:
                        if entity_score is not None and p in entity_score["properties"]:
                            p_found_in_candidate = True
                            break
                    for literal_score in scores["literal_scores"]:
                        if literal_score is not None and p in literal_score["properties"]:
                            p_found_in_candidate = True
                            break
                    if p_found_in_candidate:
                        p_found = True
                        prev_cnt = cnt
                        best_candidates.append(scores)

            if len(best_candidates) == 1:
                chosen_candidates.append(
                    {
                        "score": best_candidates[0]["score"],
                        "best_possible_score": len(best_candidates[0]["entity_scores"])
                        + len(best_candidates[0]["literal_scores"]),
                        "candidate": best_candidates[0]["candidate"],
                        "cpa_scores": get_cpa_scores(best_candidates[0]),
                        "reason": "best properties",
                    }
                )
                continue
            elif len(best_candidates) == 0:
                # If we found no candidates, use ML to choose the best one
                # TODO: ML
                # For now, just choose the first one
                chosen_candidates.append(
                    {
                        "score": candidate_scores[0]["score"],
                        "best_possible_score": len(candidate_scores[0]["entity_scores"])
                        + len(candidate_scores[0]["literal_scores"]),
                        "candidate": candidate_scores[0]["candidate"],
                        "cpa_scores": get_cpa_scores(candidate_scores[0]),
                        "reason": "no best",
                    }
                )
            else:
                # If we found multiple or zero candidates, use ML to choose the best one
                # TODO: ML
                # For now, just choose the first one
                chosen_candidates.append(
                    {
                        "score": best_candidates[0]["score"],
                        "best_possible_score": len(best_candidates[0]["entity_scores"])
                        + len(best_candidates[0]["literal_scores"]),
                        "candidate": best_candidates[0]["candidate"],
                        "cpa_scores": get_cpa_scores(best_candidates[0]),
                        "reason": "ML",
                    }
                )

        # CPA
        cpa_predictions = {}
        for from_i, to_i in self.targets["cpa"]:
            assert from_i == 0
            prop_occurrences = []
            for chosen_candidate in chosen_candidates:
                cpa_scores = chosen_candidate.get("cpa_scores")
                if cpa_scores is None or to_i not in cpa_scores:
                    continue
                properties = cpa_scores[to_i]["properties"]
                score = cpa_scores[to_i]["score"]
                if score == 0:
                    continue
                for prop in properties:
                    prop_found = False
                    for i, (p, _, _) in enumerate(prop_occurrences):
                        if p == prop:
                            prop_occurrences[i][1] += 1
                            if score > prop_occurrences[i][2]:
                                prop_occurrences[i][2] = score
                            prop_found = True
                            break
                    if not prop_found:
                        prop_occurrences.append([prop, 1, score])
            sorted_prop_occurrences = sorted(prop_occurrences, key=lambda x: (x[1], x[2]), reverse=True)
            if len(sorted_prop_occurrences) == 0:
                continue
            prop, _, score = sorted_prop_occurrences[0]
            cpa_predictions[to_i] = {"property": prop, "confidence": score}

        # Find CEA for non-subject columns
        cea_predictions = []
        for i, chosen_candidate in enumerate(chosen_candidates):
            non_subj_cea_targets = [
                (row, col) for row, col in self.targets["cea"] if row == i + 1 and col != SUBJECT_COL_INDEX
            ]
            if len(non_subj_cea_targets) == 0:
                continue

            # If candidate is None, use ML to choose candidates for non_subj_cea_targets
            if chosen_candidate["candidate"] is None:
                continue

            # ELse if ALL cpa_predictions are in the chosen candidates cpa_scores, use those
            all_predictions_found = True
            for p_id, prediction in cpa_predictions.items():
                if p_id not in chosen_candidate["cpa_scores"]:
                    all_predictions_found = False
                    break

            if all_predictions_found:
                # subject cell
                cea_predictions.append(
                    [i + 1, SUBJECT_COL_INDEX, f"http://www.wikidata.org/entity/Q{chosen_candidate['candidate'].id}"]
                )

                # non-subject cells
                for col, cpa_score in chosen_candidate["cpa_scores"].items():
                    if not (i + 1, col) in non_subj_cea_targets:
                        continue
                    assert len(cpa_score["objects"]) != 0
                    if len(cpa_score["objects"]) > 1:
                        # find index of prediction and use that to pick the object
                        predicted_prop = cpa_predictions[col]["property"]
                        for j, prop in enumerate(cpa_score["properties"]):
                            if prop == predicted_prop:
                                cea_predictions.append(
                                    [i + 1, col, f"http://www.wikidata.org/entity/Q{cpa_score['objects'][j]}"]
                                )
                                break
                    else:
                        cea_predictions.append(
                            [i + 1, col, f"http://www.wikidata.org/entity/Q{cpa_score['objects'][0]}"]
                        )
                continue

            # Else, use ML to choose candidates for non_subj_cea_targets
            x = 1

        # Gather all predictions
        cpa_preds = [
            [SUBJECT_COL_INDEX, col, f"http://www.wikidata.org/prop/direct/P{prediction['property']}"]
            for col, prediction in cpa_predictions.items()
        ]

        return cea_predictions, cpa_preds


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
