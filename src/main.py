from classes import Column
from tqdm import tqdm
from util import (
    ensemble_gradient_boost_regression,
    ensemble_hist_gradient_boost_regression,
    open_dataset,
    pickle_load,
    pickle_save,
)


# ----- Open dataset -----
print("Opening dataset...")
# cols = open_dataset(use_test_data=True)
cols: list[Column] = pickle_load("13-03_11-17-31")

# ----- Preprocess dataset -----
print("Preprocessing dataset...")
for col in tqdm(cols):
    col.get_spellchecked_mentions()
pickle_save(cols)

# ----- Fetch candidates -----
print("Fetching candidates...")
for col in tqdm(cols):
    if col.all_cells_fetched():
        continue
    col.fetch_cells()
    pickle_save(cols)


# ----- Generate features -----
print("Generating features...")
i = 1
for col in tqdm(cols):
    if col.features_fetched:
        continue
    col.compute_features()
    pickle_save(cols, f"cols-features-{i}")
    i = i + 1 if i < 9 else 1

features = []
labels = []
print("Generating feature vectors...")
for col in tqdm(cols):
    feature_vector = col.feature_vectors()
    label_vector = col.label_vectors()
    features.extend(feature_vector)
    labels.extend(label_vector)


# ----- Train regressor -----
print("Training model...")
model = ensemble_hist_gradient_boost_regression(features, labels)


# ----- Evaluate regressor -----
# max_id = max(features, key=lambda x: x[0])[0]
# for feature in features:
#     feature[0] = feature[0] / max_id

# total_correct = 0
# total_incorrect = 0

# for col in tqdm(cols):
#     for cell in col.cells:
#         cell_features = []
#         for candidate in cell.candidates:
#             feature = next(feature for feature in features if feature[0] == candidate.id)
#             cell_features.append(feature)

#         if len(cell_features) == 0:
#             continue

#         # cp = []
#         # for f in cell_features:
#         #     cp.append(sum(f[1:]))

#         # cell_predictions = list(zip(cp, cell.candidates))
#         # cell_predictions.sort(key=lambda x: x[0], reverse=True)

#         cell_predictions = model.predict(cell_features)
#         cell_predictions = list(zip(cell_predictions, cell.candidates))
#         cell_predictions.sort(key=lambda x: x[0], reverse=True)

#         # print(f"Predictions for mention: '{cell.mention}'")
#         # print(f"Candidates:")
#         # for pred in cell_predictions:
#         #     print(f"    {pred[0]}: {pred[1].title} ({pred[1].description})")

#         # print(f"Prediction:")
#         if cell_predictions[0][1].id == cell.correct_candidate.id:
#             # print(f"    CORRECT: {cell_predictions[0][1].title} ({cell_predictions[0][1].description})")
#             total_correct += 1
#         else:
#             try:
#                 correct_candidate = next(
#                     candidate for candidate in cell.candidates if candidate.id == cell.correct_candidate.id
#                 )
#                 # print(f"    INCORRECT - Should have been: {correct_candidate.title} ({correct_candidate.description})")
#             except StopIteration:
#                 # print(f"    INCORRECT - No correct candidate found :)")
#                 pass
#             total_incorrect += 1

#         # print()
#         # print()

# print(f"Total correct: {total_correct}")
# print(f"Total incorrect: {total_incorrect}")
# print(f"Accuracy: {total_correct / (total_correct + total_incorrect)}")
