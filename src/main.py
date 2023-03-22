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
PICKLE_FILE_NAME = "validation-2022"

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

model = ensemble_gradient_boost_regression(features, labels)

# ----- Evaluate model -----
precision, recall, f1 = evaluate_model(model, cols)


print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1: {f1}")
