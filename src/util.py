import csv
from typing import Union
from nltk.corpus import stopwords
import string
from sklearn.datasets import make_regression

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, explained_variance_score, mean_squared_error
from fractions import Fraction
from sklearn.tree import export_text
import pickle
from datetime import datetime


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


def random_forest_regression(data: list, labels: list[float], test_size: float = 0.3):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=test_size
    )

    X_train, y_train = make_regression(
        n_features=4, n_informative=2, random_state=42, shuffle=False
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

    # # Create a Random Forest Regressor with 100 trees
    # rf = RandomForestRegressor(n_estimators=100)

    # # Train the model on the training set
    # rf.fit(X_train, y_train)

    # Use the model to make predictions on the testing set
    # y_pred = rf.predict(X_test)

    # # Evaluate the accuracy of the model
    # variance_score = explained_variance_score(y_test, y_pred)

    # # Get the decision rules for every tree in the forest
    # for i, tree in enumerate(rf.estimators_):
    #     print(f"Tree {i + 1}")
    #     print(
    #         export_text(
    #             tree,
    #             feature_names=[
    #                 "lexscore",
    #                 "instance overlap",
    #                 "subclass overlap",
    #                 "desc overlap",
    #             ],
    #         )
    #     )
    # print(f"Variance score: {variance_score:.2f}")


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

    vals = get_csv_lines("./datasets/spellCheck/vals_labeled.csv")
    if correct_spelling:
        return [(line[1], line[2]) for line in vals]
    else:
        return [(line[0], line[2]) for line in vals]


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
    for claim in entity_data["claims"].values():
        try:
            if (
                claim[0]["mainsnak"]["snaktype"] == "novalue"
                or claim[0]["mainsnak"]["snaktype"] == "somevalue"
                or claim[0]["mainsnak"]["datatype"] != "wikibase-item"
            ):
                continue

            prop = claim[0]["mainsnak"]["property"]
            target = claim[0]["mainsnak"]["datavalue"]["value"]["id"]
            properties.append((prop, target))
        except KeyError:
            continue
        except:
            print("----- ERROR -------")
            print(claim)

    return properties


def pickle_save(obj):
    now = datetime.now()
    filename = f"src/pickle-dumps/{now.strftime('%d-%m_%H-%M-%S')}.pickle"
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def pickle_load(filename):
    file = f"src/pickle-dumps/{filename}.pickle"
    with open(file, "rb") as f:
        return pickle.load(f)
