from tqdm import tqdm
from util import open_dataset, pickle_save, pickle_load

i = 0
PICKLE_FILE_NAME = "validation"

# ----- Open dataset -----
print("Opening dataset...")
cols = open_dataset(dataset="validation", disable_spellcheck=True)
pickle_save(cols, f"{PICKLE_FILE_NAME}-{i}")
i = i + 1 if i < 9 else 1
# cols = pickle_load(f"{PICKLE_FILE_NAME}", is_dump=True)

# ----- Fetch candidates -----
print("Fetching candidates...")
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

x = 1
