from _requests import wikidata_entity_search, wikidata_get_entity, RateLimitException
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

    def __init__(self, id: int):
        self.id = id
        self.title = None
        self.description = None
        self.instances = None
        self.subclasses = None

    def fetch_info(self):
        entity_data = wikidata_get_entity(self.id)
        self.title = parse_entity_title(entity_data) or ""
        self.description = parse_entity_description(entity_data) or ""

        properties = parse_entity_properties(entity_data)
        self.instances = [int(prop[1][1:]) for prop in properties if prop[0] == "P31"]
        self.subclasses = [int(prop[1][1:]) for prop in properties if prop[0] == "P279"]

    def lex_score(self, other: str) -> float:
        return Levenshtein.ratio(self.title, other)

    def instance_overlap(self, other: "Candidate") -> tuple[int, int]:
        return (len(set(self.instances).intersection(other.instances)), len(other.instances))

    def subclass_overlap(self, other: "Candidate") -> tuple[int, int]:
        return (len(set(self.subclasses).intersection(other.subclasses)), len(other.subclasses))

    def description_overlap(self, other: "Candidate"):
        if len(self.description) == 0 or len(other.description) == 0:
            return 0.0
        vectorizer = CountVectorizer().fit_transform(
            [remove_stopwords(self.description), remove_stopwords(other.description)]
        )
        cosine_sim = cosine_similarity(vectorizer)
        return cosine_sim[0][1]
    
    def info_fetched(self) -> bool:
        return self.title is not None


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

    def fetch_candidates(self):
        if self.mention == "":
            self.candidates = []
            return

        if self.candidates is not None:
            return

        entity_ids = wikidata_entity_search(self.mention)

        candidates = []
        for entity_id in entity_ids:
            candidates.append(Candidate(int(entity_id[1:])))
        self.candidates = candidates

    def fetch_candidate_info(self):
        for candidate in self.candidates:
            if candidate.title is None:
                candidate.fetch_info()

    def fetch_correct_candidate(self) -> None:
        if self.correct_id is None or self.correct_candidate is not None:
            return

        for candidate in self.candidates:
            if candidate.id == self.correct_id:
                self.correct_candidate = candidate
                return

        self.correct_candidate = Candidate(self.correct_id)
        self.correct_candidate.fetch_info()
    
    def all_candidates_fetched(self) -> bool:
        if self.candidates is None:
            return False

        for candidate in self.candidates:
            if not candidate.info_fetched():
                return False
        
        if not self.correct_candidate.info_fetched():
            return False

        return True

    def pretty_print(self):
        print(f"Results for '{self.mention}':")
        for candidate in self.candidates:
            print(f"{candidate.id}: {candidate.title} ({candidate.description})")
            print(f"Instances: {candidate.instances}")
            print(f"Subclasses: {candidate.subclasses}")
            print()


class Column:
    cells: list[CandidateSet]

    def __init__(self):
        self.cells = []

    def add_cell(self, cell: CandidateSet):
        self.cells.append(cell)

    def fetch_cells(self):
        def fetch_worker(cell: CandidateSet):
            try:
                cell.fetch_candidates()
                cell.fetch_candidate_info()
                cell.fetch_correct_candidate()
                return
            except RateLimitException:
                return

        threads = []
        for cell in self.cells:
            t = threading.Thread(target=fetch_worker, args=[cell])
            t.start()
            threads.append(t)

        for t in threads:
            t.join()
    
    def all_cells_fetched(self) -> bool:
        for cell in self.cells:
            if not cell.all_candidates_fetched():
                return False
        return True
