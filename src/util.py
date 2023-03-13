import csv
import os
from typing import Union
from nltk.corpus import stopwords
import string
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import pickle
from datetime import datetime


def ensemble_hist_gradient_boost_regression(data, labels, test_size=0.3):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=42)

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

    mse = mean_squared_error(y_test, hgb.predict(X_test))
    print("The mean squared error (MSE) on test set: {:.4f}".format(mse))

    # Cross validation
    scores = cross_val_score(hgb, X_train, y_train, cv=5, scoring="neg_mean_squared_error")
    print("Cross-validated scores:", scores)


def ensemble_gradient_boost_regression(data, labels, test_size=0.3):
    """
    Trains a Gradient Boosting ensemble on the input data and labels and prints the accuracy of the model.

    Parameters:
    data (list): A list of input data.
    labels (list): A list of boolean labels corresponding to the input data.
    test_size (float, optional): The proportion of the data to use for testing. Default is 0.3.
    n_estimators (int, optional): The number of trees in the Gradient Boosting ensemble. Default is 100.

    Returns:
    None
    """

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=42)

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

    mse = mean_squared_error(y_test, gb.predict(X_test))
    print("The mean squared error (MSE) on test set: {:.4f}".format(mse))

    # cross validation
    scores = cross_val_score(gb, X_train, y_train, cv=5, scoring="neg_mean_squared_error")
    print("Cross-validated scores:", scores)


def random_forest_regression(data: list, labels: list[float], test_size: float = 0.3):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size)
    rf = RandomForestRegressor(
        n_estimators=500,
        min_samples_split=6,
        max_depth=4,
        criterion="squared_error",
        random_state=42,
    )
    rf.fit(X_train, y_train)

    return rf

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
    filtered_words = [word for word in filtered_words.split() if word.lower() not in stop_words]
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
        "./datasets/HardTablesR1/DataSets/HardTablesR1/Test"
        if use_test_data
        else "./datasets/HardTablesR1/DataSets/HardTablesR1/Valid"
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


def pickle_save(obj, filename: Union[str, None] = None):
    if os.path.isdir("./src/pickle-dumps") == False:
        os.mkdir("./src/pickle-dumps")

    if filename is not None:
        filename = f"src/pickle-dumps/{filename}.pickle"
    else:
        now = datetime.now()
        filename = f"src/pickle-dumps/{now.strftime('%d-%m_%H-%M-%S')}.pickle"
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
