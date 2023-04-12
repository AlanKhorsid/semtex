from collections import defaultdict
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
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))
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
    tag: Union[str, None]
    tag_ratio: Union[float, None]
    most_similar_to: Union[str, None]
    similarity_avg: Union[float, None]

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
        self.tag = None
        self.tag_ratio = None
        self.most_similar_to = None
        self.similarity_avg = None

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
    def get_tag(self) -> str:
        sentence = Sentence(self.to_sentence)
        tagger.predict(sentence)
        self.tag = ""
        # only take the first tag
        for entity in sentence.get_spans("ner"):
            self.tag = entity.tag
            break

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
        if not hasattr(self, "tag"):
            self.get_tag
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
            self.most_similar_to,
            self.similarity_avg
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
    def get_tag_ratio(self):
        # Pre-calculate named entities for all candidates
        candidate_tags = {}
        for cell in self.cells:
            for candidate in cell.candidates:
                candidate.get_tag
                candidate_tags[candidate] = candidate.tag

        # Initialize overlap count and total candidates count dictionaries
        overlap_counts = {candidate: 0 for candidate in candidate_tags}
        total_counts = {candidate: 0 for candidate in candidate_tags}

        # Compare each candidate with other candidates in different cells
        for cell in self.cells:
            for candidate in cell.candidates:
                my_tag = candidate_tags[candidate]
                for other_cell in self.cells:
                    if other_cell.mention == cell.mention:
                        continue
                    for other_cand in other_cell.candidates:
                        other_tag = candidate_tags[other_cand]
                        if my_tag == other_tag:
                            overlap_counts[candidate] += 1
                        total_counts[candidate] += 1

        # Calculate the overlap ratio for each candidate
        for cell in self.cells:
            for candidate in cell.candidates:
                if total_counts[candidate] == 0:
                    candidate.tag_ratio = 0.0
                else:
                    candidate.tag_ratio = (
                        overlap_counts[candidate] / total_counts[candidate]
                    )

    @property
    def find_most_similar(self):
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words("english"))

        def preprocess_sentence(sentence):
            words = word_tokenize(sentence)
            words = [word for word in words if not word.lower() in stop_words]
            words = [lemmatizer.lemmatize(word) for word in words]
            synsets = [wn.synsets(word) for word in words]
            synsets = [synset for sublist in synsets for synset in sublist]
            return synsets

        preprocessed_sentences = defaultdict(dict)

        for cell in self.cells:
            for candidate in cell.candidates:
                candidate.most_similar_to = ""
                if candidate not in preprocessed_sentences[cell.mention]:
                    preprocessed_sentences[cell.mention][
                        candidate
                    ] = preprocess_sentence(candidate.to_sentence)

        for cell in self.cells:
            for candidate in cell.candidates:
                my_synsets = preprocessed_sentences[cell.mention][candidate]

                most_similar_candidate = None
                candidate.similarity_avg = 0
                for other_cell in self.cells:
                    max_similarity = -1
                    if other_cell.mention == cell.mention:
                        continue

                    for other_candidate in other_cell.candidates:
                        similarity_score = 0
                        if (
                            other_candidate
                            not in preprocessed_sentences[other_cell.mention]
                        ):
                            preprocessed_sentences[other_cell.mention][
                                other_candidate
                            ] = preprocess_sentence(other_candidate.to_sentence)

                        other_synsets = preprocessed_sentences[other_cell.mention][
                            other_candidate
                        ]
                        similarity_sum = sum(
                            ms.path_similarity(synset) or 0
                            for ms in my_synsets
                            for synset in other_synsets
                        )

                        similarity_score = similarity_sum / (
                            len(my_synsets) * len(other_synsets)
                        )

                        if similarity_score > max_similarity:
                            max_similarity = similarity_score
                            most_similar_candidate = other_candidate

                    if candidate.most_similar_to == "":
                        candidate.most_similar_to += most_similar_candidate.to_sentence
                        candidate.similarity_avg += max_similarity
                    else:
                        candidate.most_similar_to += (
                            " | " + most_similar_candidate.to_sentence
                        )
                        candidate.similarity_avg += max_similarity

                if len(self.cells) == 1:
                    candidate.similarity_avg = 0
                else:
                    candidate.similarity_avg /= len(self.cells) - 1

                GREEN = "\033[92m"
                RESET = "\033[0m"

                print(f"{candidate.to_sentence} is most similar to these candidates:")
                print(candidate.most_similar_to)
                print(
                    f"Average similarity score: {GREEN}{candidate.similarity_avg}{RESET}"
                )
                print("---------------------------------")
                print()
