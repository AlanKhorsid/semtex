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
from flair.data import Sentence
from flair.models import SequenceTagger

tagger = SequenceTagger.load("flair/ner-english-ontonotes-large")


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
    sentence: Union[str, None]
    named_entity: Union[str, None]

    instance_overlap_l1_l2: Union[int, None]
    instance_overlap_l2_l1: Union[int, None]
    instance_overlap_l2_l2: Union[int, None]
    subclass_overlap_l1_l2: Union[int, None]
    subclass_overlap_l2_l1: Union[int, None]
    subclass_overlap_l2_l2: Union[int, None]
    instance_overlap_l1_l3: Union[int, None]
    instance_overlap_l2_l3: Union[int, None]
    instance_overlap_l3_l1: Union[int, None]
    instance_overlap_l3_l2: Union[int, None]
    instance_overlap_l3_l3: Union[int, None]
    subclass_overlap_l1_l3: Union[int, None]
    subclass_overlap_l2_l3: Union[int, None]
    subclass_overlap_l3_l1: Union[int, None]
    subclass_overlap_l3_l2: Union[int, None]
    subclass_overlap_l3_l3: Union[int, None]

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

        self.instance_overlap_l1_l2 = None
        self.instance_overlap_l2_l1 = None
        self.instance_overlap_l2_l2 = None
        self.subclass_overlap_l1_l2 = None
        self.subclass_overlap_l2_l1 = None
        self.subclass_overlap_l2_l2 = None
        self.instance_overlap_l1_l3 = None
        self.instance_overlap_l2_l3 = None
        self.instance_overlap_l3_l1 = None
        self.instance_overlap_l3_l2 = None
        self.instance_overlap_l3_l3 = None
        self.subclass_overlap_l1_l3 = None
        self.subclass_overlap_l2_l3 = None
        self.subclass_overlap_l3_l1 = None
        self.subclass_overlap_l3_l2 = None
        self.subclass_overlap_l3_l3 = None

    @property
    def to_sentence(self) -> str:
        x = ""

        if self.title != "":
            x += f"{self.title}."
        if self.description != "":
            x += f" {self.description}."

        props = wikidata_get_entity(self.id)["properties"]
        for prop in props:
            if prop[0] != "P31" and prop[0] != "P279":
                continue
            prop_title = wikidata_get_entity(int(prop[1][1:]))["title"]
            x += f" {prop_title}."

        return x

    @property
    def get_named_entity(self) -> str:
        sentence = Sentence(self.to_sentence)
        tagger.predict(sentence)
        for entity in sentence.get_spans("ner"):
            # print(sentence, entity.tag, entity.score)
            return entity.tag

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

    def compute_features_l2(
        self,
        other_instances_l1: list[int],
        other_subclasses_l1: list[int],
        other_instances_l2: list[int],
        other_subclasses_l2: list[int],
    ):
        (my_instances_l2, my_subclasses_2) = self.instance_layer_2
        self.instance_overlap_l1_l2 = sum(
            [self.instances.count(num) for num in other_instances_l2]
        )
        self.instance_overlap_l2_l1 = sum(
            [my_instances_l2.count(num) for num in other_instances_l1]
        )
        self.instance_overlap_l2_l2 = sum(
            [my_instances_l2.count(num) for num in other_instances_l2]
        )
        self.subclass_overlap_l1_l2 = sum(
            [self.subclasses.count(num) for num in other_subclasses_l2]
        )
        self.subclass_overlap_l2_l1 = sum(
            [my_subclasses_2.count(num) for num in other_subclasses_l1]
        )
        self.subclass_overlap_l2_l2 = sum(
            [my_subclasses_2.count(num) for num in other_subclasses_l2]
        )

    def compute_features_l3(
        self,
        other_instances_l1: list[int],
        other_instances_l2: list[int],
        other_instances_l3: list[int],
        other_subclasses_l1: list[int],
        other_subclasses_l2: list[int],
        other_subclasses_l3: list[int],
    ):
        (my_instances_l2, my_subclasses_2) = self.instance_layer_2
        (my_instances_l3, my_subclasses_3) = self.instance_layer_3
        self.instance_overlap_l1_l3 = sum(
            [self.instances.count(num) for num in other_instances_l3]
        )
        self.instance_overlap_l2_l3 = sum(
            [my_instances_l2.count(num) for num in other_instances_l3]
        )
        self.instance_overlap_l3_l1 = sum(
            [my_instances_l3.count(num) for num in other_instances_l1]
        )
        self.instance_overlap_l3_l2 = sum(
            [my_instances_l3.count(num) for num in other_instances_l2]
        )
        self.instance_overlap_l3_l3 = sum(
            [my_instances_l3.count(num) for num in other_instances_l3]
        )
        self.subclass_overlap_l1_l3 = sum(
            [self.subclasses.count(num) for num in other_subclasses_l3]
        )
        self.subclass_overlap_l2_l3 = sum(
            [my_subclasses_2.count(num) for num in other_subclasses_l3]
        )
        self.subclass_overlap_l3_l1 = sum(
            [my_subclasses_3.count(num) for num in other_subclasses_l1]
        )
        self.subclass_overlap_l3_l2 = sum(
            [my_subclasses_3.count(num) for num in other_subclasses_l2]
        )
        self.subclass_overlap_l3_l3 = sum(
            [my_subclasses_3.count(num) for num in other_subclasses_l3]
        )

    @property
    def features(self) -> list:
        return [
            self.id,
            self.title,
            self.description,
            self.num_statements,
            self.instance_overlap,
            self.subclass_overlap,
            self.description_overlap,
            self.tag,
            self.tag_ratio,
            # self.instance_names,
        ]

    @property
    def instance_names(self):
        names = [wikidata_get_entity(i)["title"] for i in self.instances]
        # create string split by |
        return "|".join(names)

    @property
    def instance_layer_2(self):
        instances = set()
        subclasses = set()
        for i in self.instances + self.subclasses:
            props = wikidata_get_entity(i)["properties"]
            ins = [int(prop[1][1:]) for prop in props if prop[0] == "P31"]
            sub = [int(prop[1][1:]) for prop in props if prop[0] == "P279"]
            instances.update(ins)
            subclasses.update(sub)
        return list(instances), list(subclasses)

    @property
    def instance_layer_3(self):
        instances = set()
        subclasses = set()
        for i in self.instances + self.subclasses:
            props = wikidata_get_entity(i)["properties"]
            ins = [int(prop[1][1:]) for prop in props if prop[0] == "P31"]
            sub = [int(prop[1][1:]) for prop in props if prop[0] == "P279"]
            for j in ins + sub:
                props = wikidata_get_entity(j)["properties"]
                ins1 = [int(prop[1][1:]) for prop in props if prop[0] == "P31"]
                sub1 = [int(prop[1][1:]) for prop in props if prop[0] == "P279"]
                instances.update(ins1)
                subclasses.update(sub1)
        return list(instances), list(subclasses)


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

    def add_layer(self, col: "Column"):
        other_candidates: list[Candidate] = []
        for cell in col.cells:
            if (cell.correct_id is not None and cell.correct_id != self.correct_id) or (
                cell.mention != self.mention
            ):
                other_candidates.extend(cell.candidates)

        l1_instances = []
        l1_subclasses = []
        l2_instances = []
        l2_subclasses = []
        l3_instances = []
        l3_subclasses = []
        for candidate in other_candidates:
            (ins_l2, sub_l2) = candidate.instance_layer_2
            (ins_l3, sub_l3) = candidate.instance_layer_3
            l1_instances.extend(candidate.instances)
            l1_subclasses.extend(candidate.subclasses)
            l2_instances.extend(ins_l2)
            l2_subclasses.extend(sub_l2)
            l3_instances.extend(ins_l3)
            l3_subclasses.extend(sub_l3)

        for candidate in self.candidates:
            candidate.compute_features_l3(
                l1_instances,
                l1_subclasses,
                l2_instances,
                l2_subclasses,
                l3_instances,
                l3_subclasses,
            )

    @property
    def features(self) -> list[list]:
        features = []
        for candidate in self.candidates:
            f = candidate.features
            f.append(Levenshtein.ratio(candidate.title, self.mention))
            f.append(1 if candidate.id == self.correct_id else -1)
            features.append(f)
        return features

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

    @property
    def get_tag_ratio(self) -> float:
        # Pre-calculate named entities for all candidates
        for cell in self.cells:
            for candidate in cell.candidates:
                candidate.get_named_entity

            for candidate in cell.candidates:
                # Calculate overlap ratio for this candidate
                overlap_counter = 0
                num_other_candidates = 0
                my_tag = candidate.get_named_entity
                # print(f"{my_tag}:  ({candidate.to_sentence})")
                other_tag = ""
                for other_cand in cell.candidates:
                    if other_cand == candidate:
                        continue
                    # print(f"{other_tag}:  ({other_cand.to_sentence})")
                    other_tag = other_cand.get_named_entity
                    if my_tag == other_tag:
                        overlap_counter += 1
                    num_other_candidates += 1

                overlap_ratio = overlap_counter / num_other_candidates
                candidate.tag_ratio = overlap_ratio
