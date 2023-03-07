from tqdm import tqdm
from classes import Candidate, CandidateSet
from util import (
    ensemble_gradient_boost_regression,
    flatten_list,
    generate_features,
    open_dataset,
    pickle_load,
    pickle_save,
    random_forest_regression,
)

# ----- Open dataset -----
# dataset = open_dataset(correct_spelling=True)

# # ----- Fetch candidates -----
# candidate_sets: list[CandidateSet] = []
# for mention, id in tqdm(dataset):
#     candidate_set = CandidateSet(mention, correct_id=id)
#     candidate_set.fetch_candidates()
#     candidate_set.fetch_candidate_info()
#     candidate_sets.append(candidate_set)
# pickle_save(candidate_sets)

# ----- Generate features -----
# features, labels_clas, labels_regr = generate_features(candidate_sets)

features = pickle_load("all_correct-spelling_features")
labels = pickle_load("all_correct-spelling_labels")
labels_regr = pickle_load("all_correct-spelling_labels-regr")

# x = 1

# ----- Train regressor -----
# random_forest_regression(features, labels_regr)
ensemble_gradient_boost_regression(flatten_list(features), flatten_list(labels_regr))


# features = pickle_load("first-100_correct-spelling_features")
# labels = pickle_load("first-100_correct-spelling_labels-regr")
# random_forest_regression(features, labels)
