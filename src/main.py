from util import (
    ensemble_gradient_boost_regression,
    ensemble_hist_gradient_boost_regression,
    open_dataset,
    pickle_load,
    random_forest_regression,
)
from tqdm import tqdm


def flatten_list(nested_list: list[list[any]]) -> list[any]:
    """
    Takes a nested list of lists and returns a flattened list.
    Args:
        nested_list (list[list[T]]): The nested list to be flattened.
    Returns:
        list[T]: A flattened list containing all elements from the nested list.
    Example:
        >>> nested_list = [[[1],[2],[3]], [[4],[5],[6]], [[7],[8],[9]]]
        >>> flatten_list(nested_list)
        [1, 2, 3, 4, 5, 6, 7, 8, 9]
    """
    return [num for sublist1 in nested_list for sublist2 in sublist1 for num in sublist2]


# ----- Open dataset -----
cols = open_dataset(use_test_data=True)

# ----- Fetch candidates -----
# cols = cols[:1]
for col in tqdm(cols):
    col.fetch_cells()

x = 0

# # ----- Open dataset -----
# dataset = open_dataset(correct_spelling=True)

# # ----- Fetch candidates -----
# # candidate_sets: list[CandidateSet] = []
# # for mention, id in tqdm(dataset):
# #     candidate_set = CandidateSet(mention, correct_id=id)
# #     candidate_set.fetch_candidates()
# #     candidate_set.fetch_candidate_info()
# #     candidate_sets.append(candidate_set)
# # pickle_save(candidate_sets)
# cols = pickle_load("all_correct-spelling_cols")

# # ----- Generate features -----
# # features, labels_clas, labels_regr = generate_features(candidate_sets)
# features = pickle_load("all_correct-spelling_features")
# labels_clas = pickle_load("all_correct-spelling_labels")
# labels_regr = pickle_load("all_correct-spelling_labels-regr")

# features_flat = flatten_list(features)
# labels_clas_flat = flatten_list(labels_clas)
# labels_regr_flat = flatten_list(labels_regr)

# # ----- Train regressor -----
# # random_forest_regression(features, labels_regr)
# # model = ensemble_gradient_boost_regression(features_flat, labels_regr_flat)
# # model = random_forest_regression(features_flat, labels_regr_flat)
# model = ensemble_hist_gradient_boost_regression(features_flat, labels_regr_flat)

# # ----- Evaluate regressor -----

# total_correct = 0
# total_incorrect = 0

# for col in cols:
#     for cell in col:
#         cell_features = []
#         for candidate in cell.candidates:
#             feature = next(feature for feature in features_flat if feature[0] == candidate.id)
#             cell_features.append(feature)

#         cell_predictions = model.predict(cell_features)
#         cell_predictions = list(zip(cell_predictions, cell.candidates))
#         cell_predictions.sort(key=lambda x: x[0], reverse=True)

#         print(f"Predictions for mention: '{cell.mention}'")
#         print(f"Candidates:")
#         for pred in cell_predictions:
#             print(f"    {pred[0]}: {pred[1].title} ({pred[1].description})")

#         print(f"Prediction:")
#         if (cell_predictions[0][1].is_correct):
#             print(f"    CORRECT: {cell_predictions[0][1].title} ({cell_predictions[0][1].description})")
#             total_correct += 1
#         else:
#             try:
#                 correct_candidate = next(candidate for candidate in cell.candidates if candidate.is_correct)
#                 print(f"    INCORRECT - Should have been: {correct_candidate.title} ({correct_candidate.description})")
#             except StopIteration:
#                 print(f"    INCORRECT - No correct candidate found :)")
#             total_incorrect += 1

#         print()
#         print()

#         x = 1

# print(f"Total correct: {total_correct}")
# print(f"Total incorrect: {total_incorrect}")
# print(f"Accuracy: {total_correct / (total_correct + total_incorrect)}")
