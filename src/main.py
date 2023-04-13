import random
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from catboost.text_processing import Tokenizer
from sklearn.model_selection import ParameterGrid, train_test_split
from torch import rand
from classes import Column
from util import (
    evaluate_model,
    open_dataset,
    pickle_save,
    pickle_load,
    progress,
)
import pandas as pd

i = 0
PICKLE_FILE_NAME = "test-2022-bing"

# ----- Open dataset -----
# cols = open_dataset(dataset="validation", disable_spellcheck=False)
# pickle_save(cols, f"{PICKLE_FILE_NAME}-{i}")
# i = i + 1 if i < 9 else 1
# cols_test: list[Column] = pickle_load(f"{PICKLE_FILE_NAME}", is_dump=True)
# cols_validation: list[Column] = pickle_load("validation-2022-bing", is_dump=True)

cols_test_tag = pickle_load("all-test-tag", is_dump=True)
cols_validation_tag = pickle_load("all-validation-tag", is_dump=True)

# ----- Fetch candidates -----
while not all([col.all_cells_fetched for col in cols_test_tag]):
    with progress:
        for col in progress.track(cols_test_tag, description="Fetching candidates"):
            if col.all_cells_fetched:
                continue
            col.fetch_cells()
            pickle_save(cols_test_tag, f"{PICKLE_FILE_NAME}-{i}")
            i = i + 1 if i < 9 else 1

# ----- Generate features -----
with progress:
    for col in progress.track(cols_test_tag, description="Generating features"):
        if col.features_computed:
            continue
        col.compute_features()
        pickle_save(cols_test_tag, f"{PICKLE_FILE_NAME}-{i}")
        i = i + 1 if i < 9 else 1

    num_of_iterations = 0
    counter_test = 0

    for col in progress.track(
        cols_test_tag,
        description="Generating features for semantic similarities (test)",
    ):
        col.find_most_similar
        counter_test += 1
        if counter_test % 10 == 0:
            num_of_iterations += 1
            pickle_save(
                cols_test_tag[:counter_test],
                f"with-semantic-features-test-{num_of_iterations}",
            )

    num_of_iterations = 0
    counter_validation = 0

    for col in progress.track(
        cols_validation_tag,
        description="Generating features for semantic similarities (validation)",
    ):
        # skip the first 15 columns
        if counter_validation < 17:
            counter_validation += 1
            continue

        col.find_most_similar
        counter_validation += 1
        if counter_validation % 10 == 0:
            num_of_iterations += 1
            pickle_save(
                cols_validation_tag[:counter_validation],
                f"with-semantic-features-validation-{num_of_iterations}",
            )


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
features_test = []
features_validation = []
for col in progress.track(cols_test_tag, description="Training model"):
    features_test.extend(col.features)

for col in progress.track(cols_validation_tag, description="Training model"):
    features_validation.extend(col.features)

# max_id = max([i[0] for i in features])
# features = [[x[0] / max_id] + x[1:] for x in features]

# Process features

test = pd.DataFrame(
    {
        "id": [x[0] for x in features_test],
        "title": [x[1] for x in features_test],
        "description": [x[2] for x in features_test],
        "num_statements": [x[3] for x in features_test],
        "instance_overlap": [x[4] for x in features_test],
        "subclass_overlap": [x[5] for x in features_test],
        "description_overlap": [x[6] for x in features_test],
        "tag": [x[7] if x[7] is not None else "" for x in features_test],
        "tag_ratio": [x[8] for x in features_test],
        # "instance_names": [x[7] for x in features],
        "title_levenshtein": [x[9] for x in features_test],
        "label": [x[10] for x in features_test],
    }
)
train = pd.DataFrame(
    {
        "id": [x[0] for x in features_validation],
        "title": [x[1] for x in features_validation],
        "description": [x[2] for x in features_validation],
        "num_statements": [x[3] for x in features_validation],
        "instance_overlap": [x[4] for x in features_validation],
        "subclass_overlap": [x[5] for x in features_validation],
        "description_overlap": [x[6] for x in features_validation],
        "tag": [x[7] if x[7] is not None else "" for x in features_validation],
        "tag_ratio": [x[8] for x in features_validation],
        # "instance_names": [x[7] for x in features_validation],
        "title_levenshtein": [x[9] for x in features_validation],
        "label": [x[10] for x in features_validation],
    }
)


# text_features = ["title", "description", "tag"]
text_features = ["title", "description", "tag"]
# text_features = []

X_train = train.drop(["label"], axis=1)
y_train = train["label"]
X_test = test.drop(["label"], axis=1)
y_test = test["label"]

train_pool = Pool(
    X_train,
    y_train,
    text_features=text_features,
    feature_names=list(X_train),
)
test_pool = Pool(
    X_test,
    y_test,
    text_features=text_features,
    feature_names=list(X_train),
)

cb_params = {
    "iterations": [5000],
    "learning_rate": [0.01, 0.03, 0.1],
    "depth": [6, 8, 10, 12],
    "l2_leaf_reg": [1, 2, 3],
    # "loss_function": "MultiClassOneVsAll",
    "leaf_estimation_method": ["Newton"],
    "random_seed": [42],
    "verbose": [False],
    "random_strength": [10, 12, 14, 16, 18, 20],
    "bootstrap_type": ["Bayesian", "Bernoulli"],
    "early_stopping_rounds": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    "grow_policy": ["Lossguide"],
    "max_leaves": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    "min_data_in_leaf": [1, 2],
    # "task_type": "GPU",
    "tokenizers": [
        {
            "tokenizer_id": "Space",
            "delimiter": " ",
            "separator_type": "ByDelimiter",
        },
    ],
    "dictionaries": [
        {
            "dictionary_id": "Unigram",
            "max_dictionary_size": "50000",
            "gram_count": "1",
            "tokenizerId": "Space",
        },
    ],
    "feature_calcers": [
        "BoW:top_tokens_count=1000",
        "NaiveBayes",
    ],
}

# use parametergrid to create all combinations of parameters
param_grid = ParameterGrid(cb_params)
# shuffle the parameter grid
param_grid = list(param_grid)
random.shuffle(param_grid)

best_f1 = 0
num_of_iterations = 0
list_of_params = []
already_seen = pickle_load("used-parameters", is_dump=True)
# loop through the parameter combinations
for params in param_grid:
    if params in already_seen:
        continue
    num_of_iterations += 1
    print(params)
    print()
    print(f"{num_of_iterations} of {len(param_grid)}")
    # train the model
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=test_pool, verbose=100, early_stopping_rounds=100)
    list_of_params.append(params)

    if num_of_iterations % 50 == 0:
        pickle_save(
            list_of_params,
            f"used-parameters",
        )

    num_correct_annotations = 0
    num_submitted_annotations = 0
    num_ground_truth_annotations = 0
    with progress:
        t1 = progress.add_task("Columns", total=len(cols_test_tag))
        # t2 = progress.add_task("|-> Cells")

        for col in cols_test_tag:
            # progress.update(task_id=t2, total=len(col.cells))
            for cell in col.cells:
                num_ground_truth_annotations += 1
                if len(cell.candidates) == 0:
                    continue
                num_submitted_annotations += 1

                preds = model.predict_proba(cell.features)
                candidate_preds = list(zip(cell.candidates, preds))
                candidate_preds.sort(key=lambda x: x[1][1], reverse=True)

                if candidate_preds[0][0].id == cell.correct_id:
                    num_correct_annotations += 1
                # if len(candidate_preds) > 1 and candidate_preds[1][0].id == cell.correct_id:
                #     num_correct_annotations += 1
                # if len(candidate_preds) > 2 and candidate_preds[2][0].id == cell.correct_id:
                #     num_correct_annotations += 1

                # for candidate, pred in candidate_preds:
                #     print(
                #         f"{'CORRECT ' if candidate.id == cell.correct_id else '        '}{'{:.2f}'.format(pred[1] * 100)}%:\t{candidate.title}"
                #     )

            #     progress.update(task_id=t2, advance=1)
            # progress.update(task_id=t2, completed=0)
            progress.update(task_id=t1, advance=1)

    precision = (
        num_correct_annotations / num_submitted_annotations
        if num_submitted_annotations > 0
        else 0
    )
    recall = (
        num_correct_annotations / num_ground_truth_annotations
        if num_ground_truth_annotations > 0
        else 0
    )
    f1 = (
        2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    )

    # Define color codes
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    print(f"Precision:    {precision}")
    print(f"Recall:       {recall}")
    print(f"F1:           {f1}")
    print()
    print("----------------------------------------")
    print(f"{RED}{BOLD}Best F1 so far:      {RED}{best_f1}{BOLD}{RESET}")
    print("----------------------------------------")
    if f1 > best_f1:
        best_f1 = f1
        print(f"{GREEN}{BOLD}NEW BEST F1:{RESET}      {GREEN}{BOLD}{best_f1}{RESET}")
        print("----------------------------------------")
        print()
        if best_f1 > 0.797:
            # pickle f1, precision, recall and model
            pickle_save(
                {
                    "f1": f1,
                    "precision": precision,
                    "recall": recall,
                    "model": model,
                    "params": params,
                },
                "best-model-so-far",
            )
