from tqdm import tqdm
from classes import Column
from util import (
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

param_grid = {
    "n_estimators": [500, 800, 1000, 1200],
    "learning_rate": [0.01],
    "max_depth": [8, 12],
    "min_child_weight": [1],
    "subsample": [0.8],
    "gamma": [0.1, 0.2, 0.3],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_alpha": [0.1, 0.5, 1],
    "reg_lambda": [1, 1.5, 2],
}

# calculate number of combinations of parameters
# n_combinations = 1
# for key in param_grid:
#     n_combinations *= len(param_grid[key])

# print(f"Number of combinations: {n_combinations}")

# compute all combinations of parameters
# f1_prev = 0
# for n_estimators in param_grid["n_estimators"]:
#     for learning_rate in param_grid["learning_rate"]:
#         for max_depth in param_grid["max_depth"]:
#             for min_child_weight in param_grid["min_child_weight"]:
#                 for subsample in param_grid["subsample"]:
#                     for gamma in param_grid["gamma"]:
#                         for colsample_bytree in param_grid["colsample_bytree"]:
#                             for reg_alpha in param_grid["reg_alpha"]:
#                                 for reg_lambda in param_grid["reg_lambda"]:
#                                     params = {
#                                         "n_estimators": n_estimators,
#                                         "learning_rate": learning_rate,
#                                         "max_depth": max_depth,
#                                         "min_child_weight": min_child_weight,
#                                         "subsample": subsample,
#                                         "gamma": gamma,
#                                         "colsample_bytree": colsample_bytree,
#                                         "reg_alpha": reg_alpha,
#                                         "reg_lambda": reg_lambda,
#                                     }
#                                     model = ensemble_xgboost_regression(
#                                         features, labels, params
#                                     )
#                                     # ----- Evaluate model -----
#                                     precision, recall, f1 = evaluate_model(model, cols)
#                                     if f1 > f1_prev:
#                                         f1_prev = f1
#                                         print(f"Precision: {precision}")
#                                         print(f"Recall: {recall}")
#                                         print(f"F1: {f1}")
#                                         pickle_save(
#                                             {
#                                                 "model": model,
#                                                 "prediction": precision,
#                                                 "recall": recall,
#                                                 "f1": f1,
#                                             }
#                                         )
param_grid_gbr = {
    "n_estimators": [800, 1000, 1200],
    "min_samples_split": [5, 10, 15],
    "learning_rate": [0.01],
    "max_depth": [8, 10, 12],
    "max_features": ["sqrt", "log2"],
    "max_leaf_nodes": [10, 20],
    "subsample": [0.5, 0.8],
    "loss": ["squared_error", "absolute_error", "quantile"],
}

# Now for grandient boosting regression (sklearn)
f1_prev = 0
for n_estimators in param_grid_gbr["n_estimators"]:
    for min_samples_split in param_grid_gbr["min_samples_split"]:
        for learning_rate in param_grid_gbr["learning_rate"]:
            for max_depth in param_grid_gbr["max_depth"]:
                for max_features in param_grid_gbr["max_features"]:
                    for max_leaf_nodes in param_grid_gbr["max_leaf_nodes"]:
                        for subsample in param_grid_gbr["subsample"]:
                            for loss in param_grid_gbr["loss"]:
                                params = {
                                    "n_estimators": n_estimators,
                                    "min_samples_split": min_samples_split,
                                    "learning_rate": learning_rate,
                                    "max_depth": max_depth,
                                    "max_features": max_features,
                                    "max_leaf_nodes": max_leaf_nodes,
                                    "subsample": subsample,
                                    "loss": loss,
                                }
                                model = ensemble_gradient_boost_regression(
                                    features, labels, params
                                )
                                # ----- Evaluate model -----
                                precision, recall, f1 = evaluate_model(model, cols)
                                if f1 > f1_prev:
                                    f1_prev = f1
                                    print("")
                                    print(f"Precision: {precision}")
                                    print(f"Recall: {recall}")
                                    print(f"F1: {f1}")
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
