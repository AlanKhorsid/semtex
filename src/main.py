from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
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
    progress,
)
import pandas as pd

from _requests import wikidata_get_entities

i = 0
PICKLE_FILE_NAME = "test-2022-bing"

# ----- Open dataset -----
# cols = open_dataset(dataset="validation", disable_spellcheck=False)
# pickle_save(cols, f"{PICKLE_FILE_NAME}-{i}")
# i = i + 1 if i < 9 else 1
cols: list[Column] = pickle_load(f"{PICKLE_FILE_NAME}", is_dump=True)
cols_validation: list[Column] = pickle_load("validation-2022-bing", is_dump=True)


# ----- Fetch candidates -----
while not all([col.all_cells_fetched for col in cols]):
    with progress:
        for col in progress.track(cols, description="Fetching candidates"):
            if col.all_cells_fetched:
                continue
            col.fetch_cells()
            pickle_save(cols, f"{PICKLE_FILE_NAME}-{i}")
            i = i + 1 if i < 9 else 1

# ----- Generate features -----
with progress:
    for col in progress.track(cols, description="Generating features"):
        if col.features_computed:
            continue
        col.compute_features()
        pickle_save(cols, f"{PICKLE_FILE_NAME}-{i}")
        i = i + 1 if i < 9 else 1

# with progress:
#     t1 = progress.add_task("Columns", total=len(cols))
#     t2 = progress.add_task("|-> Cells")
#     for col in cols:
#         progress.update(task_id=t2, total=len(col.cells))
#         for cell in col.cells:
#             cell.add_layer(col)
#             progress.update(task_id=t2, advance=1)
#         progress.update(task_id=t2, completed=0)
#         progress.update(task_id=t1, advance=1)
# pickle_save(cols, f"{PICKLE_FILE_NAME}-l3")

# ----- Train model -----
features = []
for col in progress.track(cols_validation, description="Training model"):
    features.extend(col.features)
# max_id = max([i[0] for i in features])
# features = [[x[0] / max_id] + x[1:] for x in features]

# Process features
X = pd.DataFrame(
    {
        "id": [x[0] for x in features],
        "title": [x[1] for x in features],
        "description": [x[2] for x in features],
        "num_statements": [x[3] for x in features],
        "instance_overlap": [x[4] for x in features],
        "subclass_overlap": [x[5] for x in features],
        "description_overlap": [x[6] for x in features],
        "instance_names": [x[7] for x in features],
        "label": [x[8] for x in features],
    }
)

text_features = ["title", "description", "instance_names"]
# text_features = []

train, test = train_test_split(X, test_size=0.3, random_state=42)

X_train = train.drop(["label"], axis=1)
y_train = train["label"]
X_test = test.drop(["label"], axis=1)
y_test = test["label"]

train_pool = Pool(X_train, y_train, text_features=text_features, feature_names=list(X_train))
test_pool = Pool(X_test, y_test, text_features=text_features, feature_names=list(X_train))

cb_params = {
    "bootstrap_type": "Bernoulli",
    "depth": 4,
    "early_stopping_rounds": 10,
    "grow_policy": "Lossguide",
    "iterations": 5000,
    "l2_leaf_reg": 0.5,
    "leaf_estimation_method": "Newton",
    "learning_rate": 0.01,
    "max_leaves": 100,
    "min_data_in_leaf": 10,
    "random_seed": 42,
    "random_strength": 5,
    "verbose": False,
}

model = CatBoostRegressor(**cb_params)
model.fit(train_pool, eval_set=test_pool, verbose=100)

x = 1


#         X = data_part.drop(["rating_10"], axis=1)
#         y = data_part["rating_10"]
#         return X, y

#     X_learn, y_learn = preprocess_data_part(learn)
#     X_test, y_test = preprocess_data_part(test)

#     return X_learn, X_test, y_learn, y_test


# X_train, X_test, y_train, y_test = get_processed_rotten_tomatoes()


# self.id,
# self.num_statements,
# self.instance_overlap,
# self.subclass_overlap,
# self.description_overlap,
# self.instance_names,

# model = ensemble_catboost_regression(features, labels)
# pickle_save(model, f"{PICKLE_FILE_NAME}-l3-model")
# model = pickle_load("validation-2022-bing-l3-model", is_dump=True)

# ----- Evaluate model -----
precision, recall, f1 = evaluate_model(model, cols)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1: {f1}")
