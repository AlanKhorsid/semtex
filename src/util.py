import csv
import json
import os
import threading
import time
from typing import Literal, Union
from nltk.corpus import stopwords
import string
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
import matplotlib.pyplot as plt
from pathlib import Path
import xgboost as xgb
import catboost as cb
from sklearn.metrics import silhouette_score

ROOTPATH = Path(__file__).parent.parent


# Use clustering
def cluster_data(data, n_clusters=5):
    # Create a KMeans model with n_clusters
    model = KMeans(n_clusters=n_clusters, random_state=42)

    # Use fit_predict to cluster the dataset
    labels = model.fit_predict(data)

    # Create a DataFrame with labels and varieties as columns
    df = pd.DataFrame({"labels": labels, "varieties": labels})

    # Create crosstab: ct
    ct = pd.crosstab(df["labels"], df["varieties"])

    # print other metrics
    print("Inertia: ", model.inertia_)

    # silhouette score
    print("Silhouette Score: ", silhouette_score(data, labels))

    # Display ct
    print(ct)

    print(labels)


def ensemble_catboost_regression(data, labels, cb_params, test_size=0.3):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=test_size, random_state=42
    )

    # Create a CatBoost Regressor with n_estimators trees
    cb_model = cb.CatBoostRegressor(**cb_params)

    # Train the model on the training set
    cb_model.fit(X_train, y_train, eval_set=(X_test, y_test))

    return cb_model


def ensemble_xgboost_regression(data, labels, xgb_params, test_size=0.3):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=test_size, random_state=42
    )

    # Create an XGBoost Regressor with n_estimators trees
    xgb_model = xgb.XGBRegressor(**xgb_params)

    # Train the model on the training set
    xgb_model.fit(X_train, y_train)
    return xgb_model


def ensemble_hist_gradient_boost_regression(data, labels, test_size=0.3):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=test_size, random_state=42
    )

    # Hyperparameters for HistGradientBoostingRegressor
    hgb_params = {
        "max_iter": 100,
        "learning_rate": 0.1,
        "max_depth": 8,
        "min_samples_leaf": 5,
        "l2_regularization": 0.01,
        "random_state": 42,
    }

    # Create a HistGradientBoostingRegressor with max_iter iterations
    hgb = HistGradientBoostingRegressor(**hgb_params)

    # Train the model on the training set
    hgb.fit(X_train, y_train)

    return hgb


def ensemble_gradient_boost_regression(data, labels, gbr_params, test_size=0.3):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=test_size, random_state=42
    )

    # Train the final model with the best parameters
    best_gb = GradientBoostingRegressor(**gbr_params)
    best_gb.fit(X_train, y_train)

    return best_gb


# def ensemble_gradient_boost_regression(data, labels, test_size=0.3):
#     # Split the dataset into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(
#         data, labels, test_size=test_size, random_state=42
#     )

#     # Hyperparameters for Gradient Boosting Regressor
#     gbr_params = {
#         "n_estimators": 800,
#         "learning_rate": 0.01,
#         "subsample": 0.8,
#         "max_depth": 8,
#         "min_samples_split": 2,
#         "loss": "squared_error",
#         "random_state": 42,
#     }

#     # Create a Gradient Boosting Regressor with n_estimators trees
#     gb = GradientBoostingRegressor(**gbr_params)

#     # Train the model on the training set
#     gb.fit(X_train, y_train)

#     return gb


def random_forest_regression(data: list, labels: list[float], test_size: float = 0.3):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=test_size
    )
    rf = RandomForestRegressor(
        n_estimators=500,
        min_samples_split=6,
        max_depth=4,
        criterion="squared_error",
        random_state=42,
    )
    rf.fit(X_train, y_train)

    return rf


def plot_feature_importance(model, data):
    # Convert the data list to a DataFrame
    data_df = pd.DataFrame(data)
    # Calculate the feature importances
    feature_importances = model.feature_importances_
    # Convert the data list to a DataFrame
    data_df = pd.DataFrame(data)

    # Manually assign column names
    data_df.columns = [
        "Id",
        "Lex Score",
        "Inst. Overlap",
        "SubC. Overlap",
        "Desc. Overlap",
    ]

    # Get the names of the features
    feature_names = list(data_df.columns)

    # Plot the feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, feature_importances)
    plt.title("Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()


def remove_stopwords(unfiltered_string: str) -> str:
    """
    Filters a string by removing stop words and punctuations.

    Parameters
    ----------
    string : str
        The string to filter.

    Returns
    -------
    str
        The filtered string.

    Example
    -------
    >>> remove_stopwords("The quick brown fox jumps over the lazy dog.")
    "quick brown fox jumps lazy dog"
    """
    translator = str.maketrans("", "", string.punctuation)
    filtered_words = unfiltered_string.translate(translator)
    stop_words = set(stopwords.words("english"))
    filtered_words = [
        word for word in filtered_words.split() if word.lower() not in stop_words
    ]
    return " ".join(filtered_words)


def get_csv_lines(filename: str) -> list[list[str]]:
    """
    Reads a CSV file and returns a list of lines.

    Parameters
    ----------
    filename : str
        The path to the CSV file.

    Returns
    -------
    list[list[str]]
        A list of lines, where each line is a list of strings.

    Example
    -------
    >>> get_csv_lines("{ROOTPATH}/datasets/spellCheck/vals_labeled.csv")
    [
        ["Lincoln Township", "Lincoln Township", "Q7996268"],
        ["Stony Creek Township", "Stony Creek Township", "Q7996260"],
        ...
    ]
    """

    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        return list(reader)


def open_dataset(
    dataset: Literal["test", "validation"] = "validation",
    disable_spellcheck: bool = False,
):
    from classes import CandidateSet, Column
    from preprocessing.suggester import generate_suggestion

    if dataset == "test":
        file_path = f"{ROOTPATH}/datasets/HardTablesR1/DataSets/HardTablesR1/Test"
    elif dataset == "validation":
        file_path = f"{ROOTPATH}/datasets/HardTablesR1/DataSets/HardTablesR1/Valid"
    else:
        raise ValueError(f"Invalid dataset: {dataset}")

    gt_lines = get_csv_lines(f"{file_path}/gt/cea_gt.csv")
    gt_lines = [[l[0], int(l[1]), int(l[2]), l[3]] for l in gt_lines]

    current_filename = ""
    lines = []
    cols: list[Column] = []
    for filename, _, _, _ in tqdm(gt_lines):
        if filename == current_filename:
            continue
        current_filename = filename

        file = get_csv_lines(f"{file_path}/tables/{filename}.csv")

        lines = [l for l in gt_lines if l[0] == filename]
        lines.sort(key=lambda x: (x[2], x[1]))

        current_col = -1
        new_cols: list[Column] = []
        for _, row, col, entity_url in lines:
            if col != current_col:
                current_col = col
                new_cols.append(Column())

            mention = file[row][col]
            if not disable_spellcheck:
                mention = generate_suggestion(mention)
            entity_id = entity_url.split("/")[-1]
            new_cols[-1].add_cell(CandidateSet(mention, correct_id=entity_id))

        cols.extend(new_cols)

    return cols


def parse_entity_title(entity_data: dict) -> Union[str, None]:
    """
    Parses the title of an entity from the Wikidata API. If the entity has no
    English label, returns None.

    Parameters
    ----------
    entity_data : dict
        The entity data to parse.

    Returns
    -------
    str
        The title of the entity.

    Example
    -------
    >>> parse_entity_title({"labels": {"en": {"value": "Barack Obama"}}})
    "Barack Obama"
    """

    try:
        return entity_data["labels"]["en"]["value"]
    except KeyError:
        return None


def parse_entity_description(entity_data: dict) -> Union[str, None]:
    """
    Parses the description of an entity from the Wikidata API. If the entity has
    no English description, returns None.

    Parameters
    ----------
    entity_data : dict
        The entity data to parse.

    Returns
    -------
    str
        The description of the entity.

    Example
    -------
    >>> parse_entity_description({"descriptions": {"en": {"value": "44th President of the United States"}}})
    "44th President of the United States"
    """

    try:
        return entity_data["descriptions"]["en"]["value"]
    except KeyError:
        return None


def parse_entity_properties(entity_data: dict) -> dict:
    """
    Parses the claims of an entity from the Wikidata API.

    Parameters
    ----------
    entity_data : dict
        The entity data to parse.

    Returns
    -------
    dict
        The claims of the entity.

    Example
    -------
    >>> parse_entity_claims(entity_data)
    [
        ("P31", "Q5"),
        ("P21", "Q6581072"),
    ]
    """

    properties = []
    for claims in entity_data["claims"].values():
        for claim in claims:
            try:
                if (
                    claim["mainsnak"]["snaktype"] == "novalue"
                    or claim["mainsnak"]["snaktype"] == "somevalue"
                    or claim["mainsnak"]["datatype"] != "wikibase-item"
                ):
                    continue

                prop = claim["mainsnak"]["property"]
                target = claim["mainsnak"]["datavalue"]["value"]["id"]
                properties.append((prop, target))
            except KeyError:
                continue
            except:
                print("----- ERROR -------")
                print(claims)

    return properties


def pickle_save_in_folder(obj, folder):
    if os.path.isdir(f"{ROOTPATH}/src/pickle-dumps/{folder}") == False:
        os.mkdir(f"{ROOTPATH}/src/pickle-dumps/{folder}")

    now = datetime.now()
    filename = (
        f"{ROOTPATH}/src/pickle-dumps/{folder}/{now.strftime('%d-%m_%H-%M-%S')}.pickle"
    )

    # check if file already exists and if so, append a number to the filename
    i = 1
    while os.path.isfile(filename):
        filename = f"{ROOTPATH}/src/pickle-dumps/{folder}/{now.strftime('%d-%m_%H-%M-%S')}_{i}.pickle"
        i += 1

    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def pickle_save(obj, filename: Union[str, None] = None):
    if os.path.isdir(f"{ROOTPATH}/src/pickle-dumps") == False:
        os.mkdir(f"{ROOTPATH}/src/pickle-dumps")

    if filename is not None:
        filename = f"{ROOTPATH}/src/pickle-dumps/{filename}.pickle"
    else:
        now = datetime.now()
        filename = (
            f"{ROOTPATH}/src/pickle-dumps/{now.strftime('%d-%m_%H-%M-%S')}.pickle"
        )
        i = 1
        while os.path.isfile(filename):
            filename = f"{ROOTPATH}/src/pickle-dumps/{now.strftime('%d-%m_%H-%M-%S')}_{i}.pickle"
            i += 1

    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def pickle_load(filename, is_dump: bool = False):
    file = (
        f"{ROOTPATH}/src/{'pickle-dumps' if is_dump else 'pickles'}/{filename}.pickle"
    )
    with open(file, "rb") as f:
        return pickle.load(f)


def evaluate_model(model, columns):
    num_correct_annotations = 0
    num_submitted_annotations = 0
    num_ground_truth_annotations = 0
    for col in columns:
        for cell in col.cells:
            num_ground_truth_annotations += 1
            if len(cell.candidates) == 0:
                continue

            num_submitted_annotations += 1

            best_candidate = None
            best_score = float("-inf")
            for candidate in cell.candidates:
                prediction = model.predict([candidate.features])[0]
                if prediction > best_score:
                    best_score = prediction
                    best_candidate = candidate
            if best_candidate is None:
                raise Exception("No candidate found")
            elif best_candidate.id == cell.correct_candidate.id:
                num_correct_annotations += 1

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

    return precision, recall, f1


class JsonUpdater:
    def __init__(self, filename):
        self.filename = f"{ROOTPATH}{filename}"
        self.lock = threading.Lock()
        self.data = self.load_data()
        self.last_save_time = time.time()

    def load_data(self):
        with open(self.filename, "r") as f:
            return json.load(f)

    def update_data(self, key, value):
        with self.lock:
            self.data[key] = value
            current_time = time.time()
            if current_time - self.last_save_time > 30:
                self.save_data()
                self.last_save_time = current_time

    def delete_data(self, key):
        with self.lock:
            del self.data[key]
            self.save_data()

    def save_data(self):
        with open(self.filename, "w") as f:
            json.dump(self.data, f)
