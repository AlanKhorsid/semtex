from _requests import wikidata_entity_search, wikidata_get_entity
from util import parse_entity_description, parse_entity_properties, parse_entity_title
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import Levenshtein


class Candidate:
    id: int
    title: str
    description: str
    instances: list[int]
    subclasses: list[int]

    def __init__(self, id: str):
        self.id = int(id[1:])

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
        return (len(set(self.instances) & set(other.instances)), len(other.instances))

    def subclass_overlap(self, other: "Candidate") -> tuple[int, int]:
        return (len(set(self.subclasses) & set(other.subclasses)), len(other.subclasses))

    def description_overlap(self, other: "Candidate"):
        vectorizer = CountVectorizer().fit_transform([self.description, other.description])
        cosine_sim = cosine_similarity(vectorizer)
        return cosine_sim[0][1]


class CandidateSet:
    mention: str
    candidates: list[Candidate]

    def __init__(self, mention: str):
        self.mention = mention
        self.candidates = []

    def fetch_candidates(self):
        print(f"Fetching candidates for '{self.mention}'...")
        entity_ids = wikidata_entity_search(self.mention)
        print(f"Found {len(entity_ids)} candidates.")
        for entity_id in entity_ids:
            self.candidates.append(Candidate(entity_id))

    def fetch_candidate_info(self):
        for candidate in self.candidates:
            print(f"Fetching info for 'Q{candidate.id}'...")
            candidate.fetch_info()
            print(f"Found entity '{candidate.title}'")

    def pretty_print(self):
        print(f"Results for '{self.mention}':")
        for candidate in self.candidates:
            print(f"{candidate.id}: {candidate.title} ({candidate.description})")
            print(f"Instances: {candidate.instances}")
            print(f"Subclasses: {candidate.subclasses}")
            print()
