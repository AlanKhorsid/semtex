import csv
import os
from typing import Union
from nltk.corpus import stopwords
import string
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import export_text
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
import matplotlib.pyplot as plt
import spacy
from collections import Counter
import en_core_web_sm
from _types import SpacyTypes
from pathlib import Path

ROOTPATH = Path(__file__).parent.parent

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer


# Your custom metric function
def custom_metric(y_true, y_pred):
    # Compute your custom metric here
    # ...
    score = ...
    return score


def xgb_regression_hyperparameter_tuning(data, labels, test_size=0.3):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=test_size, random_state=42
    )

    # Create an XGBoost Regressor
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)

    # Hyperparameters for Grid Search
    param_grid = {
        "n_estimators": [500, 800, 1000],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [8, 10, 12],
        "min_child_weight": [1, 3, 5],
        "subsample": [0.6, 0.8, 1.0],
        "gamma": [0.1, 0.2, 0.3],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_alpha": [0.1, 0.5, 1],
        "reg_lambda": [1, 1.5, 2],
    }

    # Create a custom scorer using your custom metric function
    custom_scorer = make_scorer(custom_metric, greater_is_better=True)

    # Create a GridSearchCV instance with the XGBoost model and parameter grid
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=5,
        scoring=custom_scorer,
        n_jobs=-1,
        verbose=2,
    )

    # Perform Grid Search on the training set
    grid_search.fit(X_train, y_train)

    # Print the best hyperparameters
    print("Best hyperparameters:", grid_search.best_params_)

    # Train the model with the best hyperparameters on the training set
    best_xgb_model = grid_search.best_estimator_

    # Evaluate the model on the test set
    y_pred = best_xgb_model.predict(X_test)
    score = custom_metric(y_test, y_pred)
    print("Custom metric on test set: {:.4f}".format(score))

    return best_xgb_model


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


def gbr_hyperparameters_tuning(data, labels, test_size=0.3):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=test_size, random_state=42
    )

    # Create a Gradient Boosting Regressor
    gb = GradientBoostingRegressor(random_state=42)

    # Define a parameter grid for tuning
    param_grid = {
        "n_estimators": [800, 900, 1000, 1200],
        "learning_rate": [0.001, 0.01, 0.1],
        "subsample": [0.6, 0.8, 1.0],
        "max_depth": [3, 6, 8, 10],
        "min_samples_split": [2, 5, 10],
        "loss": ["squared_error", "absolute_error", "huber", "quantile"],
    }

    # Use your custom metric to evaluate the model
    scoring = make_scorer(custom_metric, greater_is_better=True)

    # Create GridSearchCV object
    grid_search = GridSearchCV(
        gb, param_grid, scoring=scoring, cv=5, n_jobs=-1, verbose=1
    )

    # Train the model on the training set
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_

    # Train the final model with the best parameters
    best_gb = GradientBoostingRegressor(**best_params, random_state=42)
    best_gb.fit(X_train, y_train)

    return best_gb


def ensemble_gradient_boost_regression(data, labels, test_size=0.3):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=test_size, random_state=42
    )

    # Hyperparameters for Gradient Boosting Regressor
    gbr_params = {
        "n_estimators": 800,
        "learning_rate": 0.01,
        "subsample": 0.8,
        "max_depth": 8,
        "min_samples_split": 2,
        "loss": "squared_error",
        "random_state": 42,
    }

    # Create a Gradient Boosting Regressor with n_estimators trees
    gb = GradientBoostingRegressor(**gbr_params)

    # Train the model on the training set
    gb.fit(X_train, y_train)

    return gb


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


def open_dataset(correct_spelling: bool = False, use_test_data: bool = False):
    """
    Opens the dataset and returns a list of (mention, id) tuples.

    Parameters
    ----------
    correct_spelling : bool, optional
        Whether to return the correctly preprocessed mentions or the mentions

    use_test_data : bool, optional
        Whether to use the test data or the validation data

    Returns
    -------
    list[Column]
    """

    from classes import CandidateSet, Column

    file_path = (
        f"{ROOTPATH}/datasets/HardTablesR1/DataSets/HardTablesR1/Test"
        if use_test_data
        else f"{ROOTPATH}/datasets/HardTablesR1/DataSets/HardTablesR1/Valid"
    )
    gt_lines = get_csv_lines(f"{file_path}/gt/cea_gt.csv")
    gt_lines = [[l[0], int(l[1]), int(l[2]), l[3]] for l in gt_lines]

    current_filename = ""
    lines = []
    cols: list[Column] = []
    for filename, _, _, _ in gt_lines:
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


def name_entity_recognition_labels(title: str, description: str) -> list[int]:
    nlp = en_core_web_sm.load()
    labels = []
    results = nlp(f"{title} - {description}")
    for r in results.ents:
        labels.append(r.label_)
    labels = list(set(labels))
    labels = [SpacyTypes[label].value for label in labels]
    return labels


# Merge two dictionaries and keep values of common keys in list
def merge_dict():
    pickle_save({**dict1, **dict2})
