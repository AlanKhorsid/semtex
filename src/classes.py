from _requests import wikidata_entity_search, wikidata_get_entity
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


class Candidate:
    id: int
    title: str
    description: str
    instances: list[int]
    subclasses: list[int]

    def __init__(self, id: int):
        self.id = id

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


class CandidateSet:
    mention: str
    candidates: list[Candidate]
    correct_candidate: Union[Candidate, None]
    correct_id: Union[int, None]

    def __init__(self, mention: str, correct_id: Union[str, None] = None):
        self.mention = mention
        self.candidates = []
        if correct_id is not None:
            self.correct_id = int(correct_id[1:])

    def fetch_candidates(self):
        # print(f"Fetching candidates for '{self.mention}'...")
        entity_ids = wikidata_entity_search(self.mention)
        # print(f"Found {len(entity_ids)} candidates.")
        for entity_id in entity_ids:
            self.candidates.append(Candidate(int(entity_id[1:])))

    def fetch_candidate_info(self):
        for candidate in self.candidates:
            # print(f"Fetching info for 'Q{candidate.id}'...")
            candidate.fetch_info()
            # print(f"Found entity '{candidate.title}'")

    def fetch_correct_candidate(self) -> None:
        if self.correct_id is None:
            return

        for candidate in self.candidates:
            if candidate.id == self.correct_id:
                self.correct_candidate = candidate
                return

        self.correct_candidate = Candidate(self.correct_id)
        self.correct_candidate.fetch_info()

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
        for cell in self.cells:
            cell.fetch_candidates()
            cell.fetch_candidate_info()
            cell.fetch_correct_candidate()
