from _requests import wikidata_entity_search, wikidata_get_entity, RateLimitException
from preprocessing.suggester import generate_suggestion
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
        self.instance_overlap = None
        self.subclass_overlap = None
        self.description_overlap = None
        self.lex_score = None

    def fetch_info(self):
        entity_data = wikidata_get_entity(self.id)
        self.title = parse_entity_title(entity_data) or ""
        self.description = parse_entity_description(entity_data) or ""

        properties = parse_entity_properties(entity_data)
        self.instances = [int(prop[1][1:]) for prop in properties if prop[0] == "P31"]
        self.subclasses = [int(prop[1][1:]) for prop in properties if prop[0] == "P279"]

    def description_overlap(self, other: "Candidate"):
        if len(self.description) == 0 or len(other.description) == 0:
            return 0.0
        vectorizer = CountVectorizer().fit_transform(
            [remove_stopwords(self.description), remove_stopwords(other.description)]
        )
        cosine_sim = cosine_similarity(vectorizer)
        return cosine_sim[0][1]

    def compute_features(
        self, correct: "Candidate", other: list["Candidate"], instance_total: int, subclass_total: int
    ):
        self.lex_score = Levenshtein.ratio(self.title, correct.title)

        instance_overlap = 0
        subclass_overlap = 0
        description_overlaps = []
        for other_candidate in other:
            if other_candidate.id == self.id:
                continue
            instance_overlap += len(set(self.instances).intersection(other_candidate.instances))
            subclass_overlap += len(set(self.subclasses).intersection(other_candidate.subclasses))
            description_overlaps.append(self.description_overlap(other_candidate))

        self.instance_overlap = instance_overlap / instance_total if instance_total > 0 else 0
        self.subclass_overlap = subclass_overlap / subclass_total if subclass_total > 0 else 0
        self.description_overlap = (
            sum(description_overlaps) / len(description_overlaps) if len(description_overlaps) > 0 else 0
        )

    def features(self):
        return [self.id, self.lex_score, self.instance_overlap, self.subclass_overlap, self.description_overlap]

    def info_fetched(self) -> bool:
        return self.title is not None


class CandidateSet:
    mention: str
    mention_spellchecked: Union[str, None]
    candidates: Union[list[Candidate], None]
    correct_candidate: Union[Candidate, None]
    correct_id: Union[int, None]

    def __init__(self, mention: str, correct_id: Union[str, None] = None):
        self.mention = mention
        self.candidates = None
        self.correct_candidate = None
        self.mention_spellchecked = None
        if correct_id is not None:
            self.correct_id = int(correct_id[1:])
        else:
            self.correct_id = None

    def get_spellchecked_mention(self):
        if self.mention_spellchecked is None:
            self.mention_spellchecked = generate_suggestion(self.mention)

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

    def compute_features(self, col: "Column"):
        other_candidates: list[Candidate] = []
        for cell in col.cells:
            if cell.correct_id != self.correct_id:
                other_candidates.extend(cell.candidates)

        instance_total = sum([len(candidate.instances) for candidate in other_candidates])
        subclass_total = sum([len(candidate.subclasses) for candidate in other_candidates])

        for candidate in self.candidates:
            candidate.compute_features(self.correct_candidate, other_candidates, instance_total, subclass_total)

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
    features_fetched: bool

    def __init__(self):
        self.cells = []
        self.features_fetched = False

    def add_cell(self, cell: CandidateSet):
        self.cells.append(cell)

    def get_spellchecked_mentions(self):
        for cell in self.cells:
            cell.get_spellchecked_mention()

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

    def compute_features(self):
        for cell in self.cells:
            cell.compute_features(self)
        self.features_fetched = True

    def feature_vectors(self):
        if not self.features_fetched:
            raise Exception("Features not yet computed!")

        vectors = []
        for cell in self.cells:
            for candidate in cell.candidates:
                vectors.append(candidate.features())
        return vectors

    def label_vectors(self):
        if not self.features_fetched:
            raise Exception("Features not yet computed!")

        labels = []
        for cell in self.cells:
            for candidate in cell.candidates:
                if candidate.id == cell.correct_id:
                    labels.append(1.0)
                else:
                    labels.append(0.0)
        return labels

    def all_cells_fetched(self) -> bool:
        for cell in self.cells:
            if not cell.all_candidates_fetched():
                return False
        return True
