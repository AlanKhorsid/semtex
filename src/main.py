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

from _requests import wikidata_get_entities

i = 0
PICKLE_FILE_NAME = "test-2022-bing"

# ----- Open dataset -----
# cols = open_dataset(dataset="validation", disable_spellcheck=False)
# pickle_save(cols, f"{PICKLE_FILE_NAME}-{i}")
# i = i + 1 if i < 9 else 1
cols: list[Column] = pickle_load(f"{PICKLE_FILE_NAME}-l2", is_dump=True)
cols_validation: list[Column] = pickle_load("validation-2022-bing-l2", is_dump=True)


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
# pickle_save(cols, f"{PICKLE_FILE_NAME}-l2")

# ----- Train model -----
features = []
labels = []
for col in progress.track(cols_validation, description="Training model"):
    features.extend(col.features)
    labels.extend(col.labels)
# max_id = max([i[0] for i in features])
# features = [[x[0] / max_id] + x[1:] for x in features]

model = ensemble_catboost_regression(features, labels)
# model = pickle_load("validation-2022-bing-l2-model", is_dump=True)
# pickle_save(model, f"{PICKLE_FILE_NAME}-l2-model")

# ----- Evaluate model -----
precision, recall, f1 = evaluate_model(model, cols)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1: {f1}")
