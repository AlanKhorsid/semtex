from pprint import pprint
from typing import Union
import requests
import threading

from tqdm import tqdm
from _types import ClaimType, Entity, Claim
from classes import Candidate, CandidateSet, thread_queue
from util import (
    ensemble_gradient_boost_regression,
    generate_features,
    open_dataset,
    pickle_load,
    pickle_save,
    random_forest_regression,
)

# ----- Open dataset -----
dataset = open_dataset(correct_spelling=True)

# ----- Fetch candidates -----
candidate_sets: list[CandidateSet] = []
for mention, id, _ in tqdm(dataset):
    candidate_set = CandidateSet(mention, correct_id=id)
    candidate_set.fetch_candidates()
    #thread_queue.join()
    candidate_set.fetch_candidate_info()
    # thread_queue.join()
    candidate_sets.append(candidate_set)
pickle_save(candidate_sets)

# ----- Generate features -----
features, labels_clas, labels_regr = generate_features(candidate_sets)

# ----- Train regressor -----
# random_forest_regression(features, labels_regr)
ensemble_gradient_boost_regression(features, labels_regr)

# features = pickle_load("first-100_correct-spelling_features")
# labels = pickle_load("first-100_correct-spelling_labels-regr")
# random_forest_regression(features, labels)
