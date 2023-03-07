import csv
import os
from typing import Union
from nltk.corpus import stopwords
import string
from sklearn.datasets import make_regression
from tqdm import tqdm
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, explained_variance_score, mean_squared_error
from sklearn.tree import export_text
import pickle
from datetime import datetime

# from classes import Candidate, CandidateSet

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor


def ensemble_hist_gradient_boost_regression(data, labels, test_size=0.3):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=test_size, random_state=42
    )

    # Hyperparameters for HistGradientBoostingRegressor
    hgb_params = {
        "max_iter": 500,
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

    mse = mean_squared_error(y_test, hgb.predict(X_test))
    print("The mean squared error (MSE) on test set: {:.4f}".format(mse))

    # Cross validation
    scores = cross_val_score(
        hgb, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
    )
    print("Cross-validated scores:", scores)


def ensemble_gradient_boost_regression(data, labels, test_size=0.3):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=test_size, random_state=42
    )

    # Hyperparameters for Gradient Boosting Regressor
    gbr_params = {
        "n_estimators": 500,
        "learning_rate": 0.01,
        "subsample": 0.5,
        "max_depth": 4,
        "min_samples_split": 6,
        "loss": "squared_error",
        "random_state": 42,
    }

    # Create a Gradient Boosting Regressor with n_estimators trees
    gb = GradientBoostingRegressor(**gbr_params)

    # Train the model on the training set
    gb.fit(X_train, y_train)

    mse = mean_squared_error(y_test, gb.predict(X_test))
    print("The mean squared error (MSE) on test set: {:.4f}".format(mse))

    # cross validation
    scores = cross_val_score(
        gb, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
    )
    print("Cross-validated scores:", scores)


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

    prediction = rf.predict(X_test)
    mse = mean_squared_error(y_test, prediction)
    print(mse)


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
    >>> get_csv_lines("./datasets/spellCheck/vals_labeled.csv")
    [
        ["Lincoln Township", "Lincoln Township", "Q7996268"],
        ["Stony Creek Township", "Stony Creek Township", "Q7996260"],
        ...
    ]
    """

    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        return list(reader)


def open_dataset(correct_spelling: bool = False) -> list[tuple[str, str]]:
    """
    Opens the dataset and returns a list of (mention, id) tuples.

    Parameters
    ----------
    correct_spelling : bool, optional
        Whether to return the correctly preprocessed mentions or the mentions

    Returns
    -------
    list[tuple[str, str]]
        A list of (mention, id) tuples.

    Example
    -------
    >>> open_dataset()
    [
        ('Lincoln Township', 'Q7996268'),
        ('Stony Creek Township', 'Q7996260'),
        ...
    ]
    """

    vals = get_csv_lines("./datasets/spellCheck/vals_labeled2.csv")
    if correct_spelling:
        return [(line[1], line[2], line[3]) for line in vals]
    else:
        return [(line[0], line[2], line[3]) for line in vals]


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


def pickle_save(obj):
    if os.path.isdir("./src/pickle-dumps") == False:
        os.mkdir("./src/pickle-dumps")

    now = datetime.now()
    filename = f"src/pickle-dumps/{now.strftime('%d-%m_%H-%M-%S')}.pickle"

    # check if file already exists and if so, append a number to the filename
    i = 1
    while os.path.isfile(filename):
        filename = f"src/pickle-dumps/{now.strftime('%d-%m_%H-%M-%S')}_{i}.pickle"
        i += 1

    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def pickle_load(filename, is_dump: bool = False):
    file = f"src/{'pickle-dumps' if is_dump else 'pickles'}/{filename}.pickle"
    with open(file, "rb") as f:
        return pickle.load(f)


def candidates_iter(candidate_sets: list[object], skip_index: int = -1) -> list[object]:
    for i, candidate_set in enumerate(candidate_sets):
        if i == skip_index:
            continue
        for candidate in candidate_set.candidates:
            yield candidate


def generate_features(
    candidate_sets: list[object],
) -> tuple[list, list[bool], list[float]]:
    total_features = []
    total_labels_clas = []
    total_labels_regr = []
    for i, candidate_set in tqdm(enumerate(candidate_sets), position=1, leave=False):
        features = []
        labels_clas = []
        labels_regr = []
        for candidate in candidate_set.candidates:
            instance_total = 0
            instance_overlap = 0
            subclass_total = 0
            subclass_overlap = 0
            description_overlaps = []

            for other_candidate in candidates_iter(candidate_sets, i):
                (overlap, total) = candidate.instance_overlap(other_candidate)
                instance_total += total
                instance_overlap += overlap

                (overlap, total) = candidate.subclass_overlap(other_candidate)
                subclass_total += total
                subclass_overlap += overlap

                description_overlaps.append(
                    candidate.description_overlap(other_candidate)
                )

            labels_clas.append(candidate.is_correct)
            labels_regr.append(1.0 if candidate.is_correct else 0.0)
            features.append(
                [
                    candidate.id,
                    candidate.lex_score(candidate_set.mention),
                    instance_overlap / instance_total if instance_total > 0 else 0,
                    subclass_overlap / subclass_total if subclass_total > 0 else 0,
                    sum(description_overlaps) / len(description_overlaps)
                    if len(description_overlaps) > 0
                    else 0,
                ]
            )
        total_features.append(features)
        total_labels_clas.append(labels_clas)
        total_labels_regr.append(labels_regr)

    return total_features, total_labels_clas, total_labels_regr


def flatten_list(nested_list: list[list[any]]) -> list[any]:
    """
    Takes a nested list of lists and returns a flattened list.

    Args:
        nested_list (list[list[T]]): The nested list to be flattened.

    Returns:
        list[T]: A flattened list containing all elements from the nested list.

    Example:
        >>> nested_list = [[[1],[2],[3]], [[4],[5],[6]], [[7],[8],[9]]]
        >>> flatten_list(nested_list)
        [1, 2, 3, 4, 5, 6, 7, 8, 9]
    """
    return [
        num for sublist1 in nested_list for sublist2 in sublist1 for num in sublist2
    ]
