from util import ensemble_hist_gradient_boost_regression, open_dataset, pickle_save, pickle_load
from rich.progress import TimeRemainingColumn, TaskProgressColumn, ProgressColumn, SpinnerColumn, TimeElapsedColumn, TextColumn, Progress, BarColumn, track

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
PICKLE_FILE_NAME = "validation20"

# ----- Open dataset -----
print("Opening dataset...")
cols = open_dataset(dataset="validation", disable_spellcheck=True)
pickle_save(cols, f"{PICKLE_FILE_NAME}-{i}")
i = i + 1 if i < 9 else 1
#cols = pickle_load(f"{PICKLE_FILE_NAME}", is_dump=True)

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

# # ----- Train model -----
# features = []
# labels = []
# for col in progress.track(cols, description="Training model"):
#     features.extend(col.features)
#     labels.extend(col.labels)
# features = [[x[0] / max([i[0] for i in features])] + x[1:] for x in features]

# model = ensemble_hist_gradient_boost_regression(features, labels)
# x = 1
