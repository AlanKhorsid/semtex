import random
from tqdm import tqdm
from classes import Column
from util import (
    ensemble_catboost_regression,
    ensemble_hist_gradient_boost_regression,
    ensemble_xgboost_regression,
    ensemble_gradient_boost_regression,
    evaluate_model,
    open_dataset,
    random_forest_regression,
    pickle_save,
    pickle_load,
)

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

# ----- Generate features -----
print("Generating features...")
for col in tqdm(cols):
    if col.features_computed:
        continue
    col.compute_features()
    pickle_save(cols, f"{PICKLE_FILE_NAME}-{i}")
    i = i + 1 if i < 9 else 1

# ----- Train model -----
print("Training model...")
features = []
labels = []
for col in tqdm(cols):
    features.extend(col.features)
    labels.extend(col.labels)
# max_id = max([i[0] for i in features])
# features = [[x[0] / max_id] + x[1:] for x in features]

from sklearn.model_selection import ParameterGrid, ParameterSampler
import random

# param_grid = {
#     "n_estimators": [200],
#     "learning_rate": [0.01, 0.1, 0.2],
#     "max_depth": [8],
#     "min_child_weight": [1],
#     "subsample": [0.5],
#     "colsample_bylevel": [0.8],
#     "colsample_bytree": [0.8],
# }


param_grid = {
    "bootstrap_type": ["Bernoulli"],
    "depth": [4],
    "early_stopping_rounds": [10],
    "grow_policy": ["Lossguide"],
    "iterations": [500],
    "l2_leaf_reg": [0.5],
    "leaf_estimation_method": ["Newton"],
    "learning_rate": [0.01],
    "max_leaves": [100],
    "min_data_in_leaf": [10],
    "task_type": ["GPU"],
    "devices": ["0:1"],
    "random_seed": [42],
    "random_strength": [5],
    "verbose": [False],
}
# calculate number of combinations of parameters
n_combinations = len(list(ParameterGrid(param_grid)))
print(f"Number of combinations: {n_combinations}")

# iterate over the random hyperparameters
f1_prev = 0
for i, param in enumerate(ParameterGrid(param_grid)):
    print()
    print(f"Training model {i + 1} with parameters:")
    print(param)
    model = ensemble_catboost_regression(features, labels, param)
    # ----- Evaluate model -----
    precision, recall, f1 = evaluate_model(model, cols2)
    if f1 > f1_prev:
        f1_prev = f1
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1: {f1}")
        if f1_prev > 0.62:
            pickle_save(
                {
                    "model": model,
                    "prediction": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )

# model = ensemble_gradient_boost_regression(features, labels)

# ----- Evaluate model -----
# precision, recall, f1 = evaluate_model(model, cols)


# print(f"Precision: {precision}")
# print(f"Recall: {recall}")
# print(f"F1: {f1}")
