from tqdm import tqdm
from util import ensemble_hist_gradient_boost_regression, open_dataset, pickle_save, pickle_load

i = 0
PICKLE_FILE_NAME = "validation20"

# ----- Open dataset -----
print("Opening dataset...")
# cols = open_dataset(dataset="validation", disable_spellcheck=True)
# cols = cols[:20]
# pickle_save(cols, f"{PICKLE_FILE_NAME}-{i}")
# i = i + 1 if i < 9 else 1
cols = pickle_load(f"{PICKLE_FILE_NAME}", is_dump=True)

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

# # ----- Train model -----
# features = []
# labels = []
# for col in tqdm(cols):
#     features.extend(col.features)
#     labels.extend(col.labels)
# features = [[x[0] / max([i[0] for i in features])] + x[1:] for x in features]

# model = ensemble_hist_gradient_boost_regression(features, labels)
# x = 1
