import csv
import os
from typing import Union
from nltk.corpus import stopwords
import string
from sklearn.datasets import make_regression
from tqdm import tqdm
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, explained_variance_score, mean_squared_error
from sklearn.tree import export_text
import pickle
from datetime import datetime
# from classes import Candidate, CandidateSet


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
        data, labels, test_size=test_size
    )

    # Create a Gradient Boosting Regressor with n_estimators trees
    gb = GradientBoostingRegressor(
        n_estimators=500,
        max_depth=4,
        min_samples_split=5,
        learning_rate=0.01,
        loss="squared_error",
    )

    # Train the model on the training set
    gb.fit(X_train, y_train)

    mse = mean_squared_error(y_test, gb.predict(X_test))
    print("The mean squared error (MSE) on test set: {:.4f}".format(mse))


def ensemble_gradient_boost_classifier(data, labels, test_size=0.3, n_estimators=200):
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
        data, labels, test_size=test_size
    )

    # Create a Gradient Boosting Classifier with n_estimators trees
    gb = GradientBoostingClassifier(
        n_estimators=n_estimators, learning_rate=0.05, subsample=0.5, max_depth=2
    )

    # Train the model on the training set
    gb.fit(X_train, y_train)

    # Use the model to make predictions on the testing set
    y_pred = gb.predict(X_test)

    score = gb.score(X_test, y_test)
    print(f"Score: {score:.2f}")


def random_forest_regression(data: list, labels: list[float], test_size: float = 0.3):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size)

    # X_train, y_train = make_regression(
    #     n_features=4, n_informative=2, random_state=0, shuffle=False
    # )
    rf = RandomForestRegressor(
        n_estimators=500, criterion="squared_error", min_samples_split=5, max_depth=4
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


def random_forest(data: list, labels: list[bool], test_size: float = 0.3):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size)

    # Create a Random Forest Classifier with 100 trees
    rf = RandomForestClassifier(n_estimators=100)

    # Train the model on the training set
    rf.fit(X_train, y_train)

    # Use the model to make predictions on the testing set
    y_pred = rf.predict(X_test)

    # Evaluate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)

    # Get the decision rules for every tree in the forest
    for i, tree in enumerate(rf.estimators_):
        print(f"Tree {i + 1}")
        print(
            export_text(
                tree,
                feature_names=[
                    "lexscore",
                    "instance overlap",
                    "subclass overlap",
                    "desc overlap",
                ],
            )
        )
    # print(f"Accuracy: {accuracy:.2f}")
    # print the score
    score = rf.score(X_test, y_test)
    print(f"Score: {score:.2f}")


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


def generate_features(candidate_sets: list[object]) -> tuple[list, list[bool], list[float]]:
    features = []
    labels_clas = []
    labels_regr = []
    for i, candidate_set in enumerate(candidate_sets):
        print(f"{i + 1}/{len(candidate_sets)} Generating features for {candidate_set.mention}")
        for candidate in tqdm(candidate_set.candidates):
            # print(f"Generating features for {candidate.title}")
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

                description_overlaps.append(candidate.description_overlap(other_candidate))

            labels_clas.append(candidate.is_correct)
            labels_regr.append(1.0 if candidate.is_correct else 0.0)
            features.append(
                [
                    candidate.lex_score(candidate_set.mention),
                    instance_overlap / instance_total if instance_total > 0 else 0,
                    subclass_overlap / subclass_total if subclass_total > 0 else 0,
                    sum(description_overlaps) / len(description_overlaps) if len(description_overlaps) > 0 else 0,
                ]
            )
    
    pickle_save(features)
    pickle_save(labels_clas)
    pickle_save(labels_regr)
    
    return features, labels_clas, labels_regr