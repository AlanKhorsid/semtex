from classes import Column
from rich.progress import TimeRemainingColumn, TaskProgressColumn, ProgressColumn, SpinnerColumn, TimeElapsedColumn, TextColumn, Progress, BarColumn, track
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

progress = Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TaskProgressColumn(),
    TextColumn("[yellow]Elapsed:"),
    TimeElapsedColumn(),
    TextColumn("[cyan]ETA:"),
    TimeRemainingColumn(),
)

i = 0
PICKLE_FILE_NAME = "test-2022-bing"

# ----- Open dataset -----
print("Opening dataset...")
# cols = open_dataset(dataset="validation", disable_spellcheck=False)
# pickle_save(cols, f"{PICKLE_FILE_NAME}-{i}")
# i = i + 1 if i < 9 else 1
cols: list[Column] = pickle_load(f"{PICKLE_FILE_NAME}", is_dump=True)

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

# ----- Train model -----
features = []
labels = []
for col in progress.track(cols, description="Training model"):
    features.extend(col.features)
    labels.extend(col.labels)
# # max_id = max([i[0] for i in features])
# # features = [[x[0] / max_id] + x[1:] for x in features]

# model = ensemble_xgboost_regression(features, labels)
model = pickle_load("model-validation-2022-bing", is_dump=True)

# ----- Evaluate model -----
precision, recall, f1 = evaluate_model(model, cols)


print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1: {f1}")
