from classes import Column
from util import (
    ensemble_gradient_boost_regression,
    ensemble_hist_gradient_boost_regression,
    open_dataset,
    pickle_load,
    pickle_save,
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
# cols = open_dataset(use_test_data=True)
cols: list[Column] = pickle_load("cols-features-9", is_dump=True)

# ----- Fetch candidates -----
for col in tqdm(cols):
    if col.all_cells_fetched():
        continue
    col.fetch_cells()
    pickle_save(cols)

# ----- Generate features -----
i = 1
for col in tqdm(cols):
    if col.features_fetched:
        continue
    col.compute_features()
    pickle_save(cols, f"cols-features-{i}")
    i = i + 1 if i < 9 else 1

features = []
labels = []
for col in tqdm(cols):
    feature_vector = col.feature_vectors()
    label_vector = col.label_vectors()
    features.extend(feature_vector)
    labels.extend(label_vector)

x = 1

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
model = pickle_load("10-03_08-26-46", is_dump=True)

# ----- Evaluate regressor -----

# max_id = max(features, key=lambda x: x[0])[0]
# for feature in features:
#     feature[0] = feature[0] / max_id

total_correct = 0
total_incorrect = 0

for col in tqdm(cols):
    for cell in col.cells:
        cell_features = []
        for candidate in cell.candidates:
            feature = next(feature for feature in features if feature[0] == candidate.id)
            cell_features.append(feature)

        if len(cell_features) == 0:
            continue

        # cp = []
        # for f in cell_features:
        #     cp.append(sum(f[1:]))

        # cell_predictions = list(zip(cp, cell.candidates))
        # cell_predictions.sort(key=lambda x: x[0], reverse=True)

        cell_predictions = model.predict(cell_features)
        cell_predictions = list(zip(cell_predictions, cell.candidates))
        cell_predictions.sort(key=lambda x: x[0], reverse=True)

        # print(f"Predictions for mention: '{cell.mention}'")
        # print(f"Candidates:")
        # for pred in cell_predictions:
        #     print(f"    {pred[0]}: {pred[1].title} ({pred[1].description})")

        # print(f"Prediction:")
        if cell_predictions[0][1].id == cell.correct_candidate.id:
            # print(f"    CORRECT: {cell_predictions[0][1].title} ({cell_predictions[0][1].description})")
            total_correct += 1
        else:
            try:
                correct_candidate = next(
                    candidate for candidate in cell.candidates if candidate.id == cell.correct_candidate.id
                )
                # print(f"    INCORRECT - Should have been: {correct_candidate.title} ({correct_candidate.description})")
            except StopIteration:
                # print(f"    INCORRECT - No correct candidate found :)")
                pass
            total_incorrect += 1

        # print()
        # print()

print(f"Total correct: {total_correct}")
print(f"Total incorrect: {total_incorrect}")
print(f"Accuracy: {total_correct / (total_correct + total_incorrect)}")
