from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from catboost.text_processing import Tokenizer
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
from pattern.en import lemma
from nltk.stem import WordNetLemmatizer

from _requests import wikidata_get_entities

i = 0
PICKLE_FILE_NAME = "test-2022-bing"

# ----- Open dataset -----
# cols = open_dataset(dataset="validation", disable_spellcheck=False)
# pickle_save(cols, f"{PICKLE_FILE_NAME}-{i}")
# i = i + 1 if i < 9 else 1
cols_test: list[Column] = pickle_load(f"{PICKLE_FILE_NAME}", is_dump=True)
cols_validation: list[Column] = pickle_load("validation-2022-bing", is_dump=True)


# ----- Fetch candidates -----
while not all([col.all_cells_fetched for col in cols_test]):
    with progress:
        for col in progress.track(cols_test, description="Fetching candidates"):
            if col.all_cells_fetched:
                continue
            col.fetch_cells()
            pickle_save(cols_test, f"{PICKLE_FILE_NAME}-{i}")
            i = i + 1 if i < 9 else 1

# ----- Generate features -----
with progress:
    for col in progress.track(cols_test, description="Generating features"):
        if col.features_computed:
            continue
        col.compute_features()
        pickle_save(cols_test, f"{PICKLE_FILE_NAME}-{i}")
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
features_test = []
features_validation = []
for col in progress.track(cols_test, description="Training model"):
    features_test.extend(col.features)

for col in progress.track(cols_validation, description="Training model"):
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
        # "instance_names": [x[7] for x in features],
        "title_levenshtein": [x[7] for x in features_test],
        "label": [x[8] for x in features_test],
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
        # "instance_names": [x[7] for x in features_test],
        "title_levenshtein": [x[7] for x in features_validation],
        "label": [x[8] for x in features_validation],
    }
)

text_features = ["title", "description"]
# text_features = []

X_train = train.drop(["label"], axis=1)
y_train = train["label"]
X_test = test.drop(["label"], axis=1)
y_test = test["label"]

# ----- Tokenize -----

tokenizer = Tokenizer(lowercasing=True, separator_type="BySense", token_types=["Word", "Number", "Punctuation"])
lemmatizer = WordNetLemmatizer()


def lemmatize_text(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])


# def lemmatize_text_gensim(text):
#     result = []
#     for token in tokenizer.tokenize(text):
#         lemmas = lemmatize(token)
#         if len(lemmas) == 0:
#             lemma = token.lower()
#         else:
#             lemma = lemmas[0].decode("utf-8").split("/")[0]

#         result.append(lemma)
#     return " ".join(result)


def preprocess_data(X):
    X_preprocessed = X.copy()
    for feature in text_features:
        X_preprocessed[feature] = X[feature].apply(lambda x: lemmatize_text(" ".join(tokenizer.tokenize(x))))
    return X_preprocessed


X_preprocessed_train = preprocess_data(X_train)
X_preprocessed_test = preprocess_data(X_test)

train_pool = Pool(X_preprocessed_train, y_train, text_features=text_features, feature_names=list(X_train))
test_pool = Pool(X_preprocessed_test, y_test, text_features=text_features, feature_names=list(X_train))

cb_params = {
    "iterations": 5000,
    "learning_rate": 0.03,
    "depth": 8,
    "l2_leaf_reg": 0.1,
    # "loss_function": "MultiClassOneVsAll",
    "leaf_estimation_method": "Newton",
    "random_seed": 42,
    "verbose": False,
    "random_strength": 8,
    "bootstrap_type": "Bayesian",
    "early_stopping_rounds": 10,
    "grow_policy": "SymmetricTree",
    "min_data_in_leaf": 1,
    "task_type": "GPU",
    # "dictionaries": ["Word:min_token_occurrence=5", "BiGram:gram_order=2"],
    # "text_processing": ["NaiveBayes+Word|BoW+Word,BiGram"],
}

model = CatBoostClassifier(**cb_params)

model.fit(train_pool, eval_set=test_pool, verbose=100, early_stopping_rounds=100)
# model.fit(train_pool, verbose=100)
# pickle_save(model, "le-classifier")
# model = pickle_load("le-classifier", is_dump=True)

# ----- Evaluate model -----
# precision, recall, f1 = evaluate_model(model, cols)

num_correct_annotations = 0
num_submitted_annotations = 0
num_ground_truth_annotations = 0
with progress:
    t1 = progress.add_task("Columns", total=len(cols_test))
    t2 = progress.add_task("|-> Cells")

    for col in cols_test:
        progress.update(task_id=t2, total=len(col.cells))
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

            progress.update(task_id=t2, advance=1)
        progress.update(task_id=t2, completed=0)
        progress.update(task_id=t1, advance=1)

precision = num_correct_annotations / num_submitted_annotations if num_submitted_annotations > 0 else 0
recall = num_correct_annotations / num_ground_truth_annotations if num_ground_truth_annotations > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1: {f1}")
