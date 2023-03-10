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

    is_correct: bool

    def __init__(self, id: str, is_correct: bool = False):
        self.id = int(id[1:])
        self.is_correct = is_correct

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
        return (
            len(set(self.instances).intersection(other.instances)),
            len(other.instances),
        )

    def subclass_overlap(self, other: "Candidate") -> tuple[int, int]:
        return (
            len(set(self.subclasses).intersection(other.subclasses)),
            len(other.subclasses),
        )

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
    correct_id: Union[str, None]

    def __init__(self, mention: str, correct_id: Union[str, None] = None):
        self.mention = mention
        self.candidates = []
        self.correct_id = correct_id

    def fetch_candidates(self):
        # print(f"Fetching candidates for '{self.mention}'...")
        entity_ids = wikidata_entity_search(self.mention)
        # print(f"Found {len(entity_ids)} candidates.")
        for entity_id in entity_ids:
            is_correct = entity_id == self.correct_id
            self.candidates.append(Candidate(entity_id, is_correct))

    def fetch_candidate_info(self):
        for candidate in self.candidates:
            # print(f"Fetching info for 'Q{candidate.id}'...")
            candidate.fetch_info()
            # print(f"Found entity '{candidate.title}'")

    def pretty_print(self):
        print(f"Results for '{self.mention}':")
        for candidate in self.candidates:
            print(f"{candidate.id}: {candidate.title} ({candidate.description})")
            print(f"Instances: {candidate.instances}")
            print(f"Subclasses: {candidate.subclasses}")
            print()
