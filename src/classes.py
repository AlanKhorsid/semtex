from _requests import wikidata_entity_search, wikidata_get_entity, RateLimitException
from suggester import generate_suggestion
from util import (
    parse_entity_description,
    parse_entity_properties,
    parse_entity_title,
    remove_stopwords,
)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Union
import Levenshtein
import threading


class Candidate:
    id: int
    title: Union[str, None]
    description: Union[str, None]
    instances: Union[list[int], None]
    subclasses: Union[list[int], None]
    num_statements: Union[int, None]

    instance_overlap: Union[int, None]
    subclass_overlap: Union[int, None]
    description_overlap: Union[float, None]
    lex_score: Union[float, None]

    def __init__(self, id: int):
        self.id = id
        self.title = None
        self.description = None
        self.instances = None
        self.subclasses = None
        self.num_statements = None
        self.instance_overlap = None
        self.subclass_overlap = None
        self.description_overlap = None
        self.lex_score = None

    @property
    def info_fetched(self) -> bool:
        return self.title is not None

    def fetch_info(self):
        entity_data = wikidata_get_entity(self.id)
        self.title = entity_data["title"]
        self.description = entity_data["description"]

        self.num_statements = len(entity_data["properties"])
        self.instances = [
            int(prop[1][1:]) for prop in entity_data["properties"] if prop[0] == "P31"
        ]
        self.subclasses = [
            int(prop[1][1:]) for prop in entity_data["properties"] if prop[0] == "P279"
        ]

    def get_description_overlap(self, other: "Candidate"):
        if len(self.description) == 0 or len(other.description) == 0:
            return 0.0
        vectorizer = CountVectorizer().fit_transform(
            [remove_stopwords(self.description), remove_stopwords(other.description)]
        )
        cosine_sim = cosine_similarity(vectorizer)
        return cosine_sim[0][1]

    def compute_features(
        self,
        correct: "Candidate",
        other: list["Candidate"],
        instance_total: int,
        subclass_total: int,
    ):
        # self.lex_score = Levenshtein.ratio(self.title, correct.title)

        instance_overlap = 0
        subclass_overlap = 0
        description_overlaps = []
        for other_candidate in other:
            if other_candidate.id == self.id:
                continue
            instance_overlap += len(
                set(self.instances).intersection(other_candidate.instances)
            )
            subclass_overlap += len(
                set(self.subclasses).intersection(other_candidate.subclasses)
            )
            description_overlaps.append(self.get_description_overlap(other_candidate))

        self.instance_overlap = (
            instance_overlap / instance_total if instance_total > 0 else 0
        )
        self.subclass_overlap = (
            subclass_overlap / subclass_total if subclass_total > 0 else 0
        )
        self.description_overlap = (
            sum(description_overlaps) / len(description_overlaps)
            if len(description_overlaps) > 0
            else 0
        )

    @property
    def features(self) -> list:
        return [
            self.id,
            self.num_statements,
            self.instance_overlap,
            self.subclass_overlap,
            self.description_overlap,
        ]


class CandidateSet:
    mention: str
    candidates: Union[list[Candidate], None]
    correct_candidate: Union[Candidate, None]
    correct_id: Union[int, None]

    def __init__(self, mention: str, correct_id: Union[str, None] = None):
        self.mention = mention
        self.candidates = None
        self.correct_candidate = None
        if correct_id is not None:
            self.correct_id = int(correct_id[1:])
        else:
            self.correct_id = None

    @property
    def candidates_fetched(self) -> bool:
        return self.candidates is not None

    def fetch_candidates(self):
        if self.mention == "":
            self.candidates = []
            return

        entity_ids = wikidata_entity_search(self.mention)

        candidates = []
        for entity_id in entity_ids:
            candidates.append(Candidate(int(entity_id[1:])))
        self.candidates = candidates

    @property
    def candidate_info_fetched(self) -> bool:
        if self.candidates is None:
            return False
        for candidate in self.candidates:
            if not candidate.info_fetched:
                return False
        return True

    def fetch_candidate_info(self):
        for candidate in self.candidates:
            if not candidate.info_fetched:
                candidate.fetch_info()

    @property
    def correct_candidate_info_fetched(self) -> bool:
        if self.correct_id is None:
            return True
        if self.correct_candidate is None:
            return False
        return self.correct_candidate.info_fetched

    def fetch_correct_candidate(self) -> None:
        for candidate in self.candidates:
            if candidate.id == self.correct_id:
                self.correct_candidate = candidate
                return

        self.correct_candidate = Candidate(self.correct_id)
        self.correct_candidate.fetch_info()

    def compute_features(self, col: "Column"):
        other_candidates: list[Candidate] = []
        for cell in col.cells:
            if (cell.correct_id is not None and cell.correct_id != self.correct_id) or (
                cell.mention != self.mention
            ):
                other_candidates.extend(cell.candidates)

        instance_total = sum(
            [len(candidate.instances) for candidate in other_candidates]
        )
        subclass_total = sum(
            [len(candidate.subclasses) for candidate in other_candidates]
        )

        for candidate in self.candidates:
            candidate.compute_features(
                self.correct_candidate, other_candidates, instance_total, subclass_total
            )

    @property
    def features(self) -> list[list]:
        return [candidate.features for candidate in self.candidates]

    @property
    def labels(self) -> list[int]:
        return [
            1.0 if candidate.id == self.correct_id else 0.0
            for candidate in self.candidates
        ]


class Column:
    cells: list[CandidateSet]
    features_computed: bool

    def __init__(self):
        self.cells = []
        self.features_computed = False

    def add_cell(self, cell: CandidateSet):
        self.cells.append(cell)

    def fetch_cells(self):
        def fetch_worker(cell: CandidateSet):
            try:
                if not cell.candidates_fetched:
                    cell.fetch_candidates()
                if not cell.candidate_info_fetched:
                    cell.fetch_candidate_info()
                if not cell.correct_candidate_info_fetched:
                    cell.fetch_correct_candidate()
            except RateLimitException:
                return

        threads = []
        for cell in self.cells:
            t = threading.Thread(target=fetch_worker, args=[cell])
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

    @property
    def all_cells_fetched(self) -> bool:
        for cell in self.cells:
            if not (
                cell.candidates_fetched
                and cell.candidate_info_fetched
                and cell.correct_candidate_info_fetched
            ):
                return False
        return True

    def compute_features(self):
        for cell in self.cells:
            cell.compute_features(self)
        self.features_computed = True

    @property
    def features(self) -> list[list]:
        if not self.features_computed:
            self.compute_features()
        return [x for cell in self.cells for x in cell.features]
        # features = []
        # for cell in self.cells:
        #     x = cell.features
        #     features.extend(x)
        # return features

    @property
    def labels(self) -> list[int]:
        if not self.features_computed:
            self.compute_features()
        return [x for cell in self.cells for x in cell.labels]
