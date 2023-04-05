import random
from tqdm import tqdm
from classes import CandidateSet, Column
from util import (
    ensemble_catboost_regression,
    evaluate_model,
    pickle_save,
    pickle_load,
)
from sklearn.model_selection import ParameterGrid, ParameterSampler
import random
from transformers import AutoTokenizer, AutoModel, RobertaTokenizer, RobertaModel

i = 0
PICKLE_FILE_NAME = "validation-2022-bing"

# ----- Open dataset -----
print("Opening dataset...")
# cols = open_dataset(dataset="validation", disable_spellcheck=False)
# pickle_save(cols, f"{PICKLE_FILE_NAME}-{i}")
# i = i + 1 if i < 9 else 1
cols: list[Column] = pickle_load(f"{PICKLE_FILE_NAME}", is_dump=True)
cols2: list[Column] = pickle_load("test-2022-bing", is_dump=True)

# ----- Fetch candidates -----
print("Fetching candidates...")
while not all([col.all_cells_fetched for col in cols]):
    for col in tqdm(cols):
        if col.all_cells_fetched:
            continue
        col.fetch_cells()
        pickle_save(cols, f"{PICKLE_FILE_NAME}-{i}")
        i = i + 1 if i < 9 else 1

# call get_named_entity() on each cell
# for col in tqdm(cols):
#     for cell in col.cells:
#         print(cell.get_named_entity)

# Pre-calculate named entities for all candidates in each cell
for col in tqdm(cols):
    for cell in col.cells:
        for candidate in cell.candidates:
            candidate.named_entity = candidate.get_named_entity()

# For every cell in a column, get all candidates
for col in tqdm(cols):
    for i, cell in enumerate(col.cells):
        my_tags = set(candidate.named_entity[1] for candidate in cell.candidates)

        overlap_counter = 0
        num_other_candidates = 0
        other_tags_sets = []
        for j, other_cell in enumerate(col.cells):
            if i == j:
                continue

            other_tags_sets.append(
                set(candidate.named_entity[1] for candidate in other_cell.candidates)
            )

        for other_tags in other_tags_sets:
            if len(my_tags.intersection(other_tags)) > 0:
                overlap_counter += 1
            num_other_candidates += len(other_tags)

        overlap_ratio = overlap_counter / num_other_candidates
        print(f"RATIO: {overlap_ratio}")

# # ----- Generate features -----
# print("Generating features...")
# for col in tqdm(cols):
#     if col.features_computed:
#         continue
#     col.compute_features()
#     pickle_save(cols, f"{PICKLE_FILE_NAME}-{i}")
#     i = i + 1 if i < 9 else 1

# # ----- Train model -----
# print("Training model...")
# features = []
# labels = []
# for col in tqdm(cols):
#     features.extend(col.features)
#     labels.extend(col.labels)
# # max_id = max([i[0] for i in features])
# # features = [[x[0] / max_id] + x[1:] for x in features]

# param_grid = {
#     "bootstrap_type": ["Bernoulli"],
#     "depth": [4],
#     "early_stopping_rounds": [10],
#     "grow_policy": ["Lossguide"],
#     "iterations": [500],
#     "l2_leaf_reg": [0.5],
#     "leaf_estimation_method": ["Newton"],
#     "learning_rate": [0.01],
#     "max_leaves": [500],
#     "min_data_in_leaf": [10],
#     "random_seed": [42],
#     "random_strength": [5],
#     "verbose": [100],
# }
# # calculate number of combinations of parameters
# n_combinations = len(list(ParameterGrid(param_grid)))
# print(f"Number of combinations: {n_combinations}")

# # iterate over the random hyperparameters
# f1_prev = 0
# for i, param in enumerate(ParameterGrid(param_grid)):
#     print()
#     print(f"Training model {i + 1} with parameters:")
#     print(param)
#     model = ensemble_catboost_regression(features, labels, param)
#     # ----- Evaluate model -----
#     precision, recall, f1 = evaluate_model(model, cols2)
#     if f1 > f1_prev:
#         f1_prev = f1
#         print(f"Precision: {precision}")
#         print(f"Recall: {recall}")
#         print(f"F1: {f1}")
#         if f1_prev > 0.63:
#             pickle_save(
#                 {
#                     "model": model,
#                     "prediction": precision,
#                     "recall": recall,
#                     "f1": f1,
#                 }
#             )


# model_name = "bert-large-uncased"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name)
# model_name = "roberta-large"
# tokenizer = RobertaTokenizer.from_pretrained(model_name)
# model = RobertaModel.from_pretrained(model_name)


# def get_embedding(text):
#     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
#     outputs = model(**inputs)
#     return outputs.last_hidden_state[:, 0, :].detach().numpy()


# print("Generating sentences...")
# num_correct = 0
# num_total = 0
# num_cols_processed = 0
# for col in cols:
#     num_cols_processed += 1
#     print(f"Processed {num_cols_processed} columns out of {len(cols)}")
#     predictions = col.get_named_entity_overlap()

#     for i, prediction in enumerate(predictions):
#         num_total += 1
#         if prediction is None:
#             continue
#         if prediction.id == col.cells[i].correct_id:
#             num_correct += 1

#     print(f"Accuracy: {num_correct / num_total}")
