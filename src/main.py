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
    progress,
)

from _requests import wikidata_get_entities

i = 0
PICKLE_FILE_NAME = "test-2022-bing"

# ----- Open dataset -----
# cols = open_dataset(dataset="validation", disable_spellcheck=False)
# pickle_save(cols, f"{PICKLE_FILE_NAME}-{i}")
# i = i + 1 if i < 9 else 1
cols: list[Column] = pickle_load(f"{PICKLE_FILE_NAME}", is_dump=True)

MAX = 50
id_cache = set()

with progress:
    t1 = progress.add_task("Columns", total=len(cols))
    t2 = progress.add_task("|-> Cells")
    t3 = progress.add_task("|--> Candidates")
    t4 = progress.add_task("|---> Entities")

    for col in cols:
        progress.update(task_id=t2, total=len(col.cells))
        for cell in col.cells:
            progress.update(task_id=t3, total=len(cell.candidates))
            for candidate in cell.candidates:
                progress.update(task_id=t4, total=len(candidate.instances + candidate.subclasses))
                for id in candidate.instances + candidate.subclasses:
                    id_cache.add(id)
                    if len(id_cache) >= MAX:
                        wikidata_get_entities(list(id_cache))
                        id_cache = set()
                    progress.update(task_id=t4, advance=1)
                progress.update(task_id=t4, completed=0)
                progress.update(task_id=t3, advance=1)
            progress.update(task_id=t3, completed=0)
            progress.update(task_id=t2, advance=1)
        progress.update(task_id=t2, completed=0)
        progress.update(task_id=t1, advance=1)


# # ----- Fetch candidates -----
# while not all([col.all_cells_fetched for col in cols]):
#     with progress:
#         for col in progress.track(cols, description="Fetching candidates"):
#             if col.all_cells_fetched:
#                 continue
#             col.fetch_cells()
#             pickle_save(cols, f"{PICKLE_FILE_NAME}-{i}")
#             i = i + 1 if i < 9 else 1

# # ----- Generate features -----
# with progress:
#     for col in progress.track(cols, description="Generating features"):
#         if col.features_computed:
#             continue
#         col.compute_features()
#         pickle_save(cols, f"{PICKLE_FILE_NAME}-{i}")
#         i = i + 1 if i < 9 else 1

# # ----- Train model -----
# features = []
# labels = []
# for col in progress.track(cols, description="Training model"):
#     features.extend(col.features)
#     labels.extend(col.labels)
# # # max_id = max([i[0] for i in features])
# # # features = [[x[0] / max_id] + x[1:] for x in features]

# model = ensemble_xgboost_regression(features, labels)
# model = pickle_load("model-validation-2022-bing", is_dump=True)

# # ----- Evaluate model -----
# precision, recall, f1 = evaluate_model(model, cols)


# print(f"Precision: {precision}")
# print(f"Recall: {recall}")
# print(f"F1: {f1}")
