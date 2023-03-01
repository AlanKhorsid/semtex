from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from fractions import Fraction
from sklearn.tree import export_text

# Candidates:
# [barack obama (Q76), obama (Q41773), obama (Q18355807)]
# [donald trump (Q22686), donald trump (Q27947481), donald trump jr. (Q3713655)]
# [vladimir putin (Q7747), putin khuylo! (Q17195494),putin (Q30524893)]
# [emmanuel macron (Q3052772), macron (Q627627), macron (Q1439645)]


class Dataset:
    def __init__(self):
        self.data = []
        self.target = []
        self.feature_names = [
            "lexscore",
            "instance overlap",
            "subclass overlap",
            "desc overlap",
        ]


presidents = [
    [7, 4 / 9, 0 / 9, 3 / 9, True],
    [0, 0 / 9, 0 / 9, 0 / 9, False],
    [0, 0 / 9, 0 / 9, 0 / 9, False],
    [0, 4 / 9, 0 / 9, 3 / 9, True],
    [0, 4 / 9, 0 / 9, 0 / 9, False],
    [4, 4 / 9, 0 / 9, 0 / 9, False],
    [9, 4 / 9, 0 / 9, 3 / 9, True],
    [8, 0 / 9, 0 / 9, 0 / 9, False],
    [0, 0 / 9, 0 / 9, 0 / 9, False],
    [9, 4 / 9, 0 / 9, 3 / 9, True],
    [0, 0 / 9, 0 / 9, 0 / 9, False],
    [0, 0 / 9, 0 / 9, 0 / 9, False],
]

dataset = Dataset()

for row in presidents:
    dataset.data.append(row[:-1])
    dataset.target.append(row[-1])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, test_size=0.3
)
print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# Create a Random Forest Classifier with 100 trees
rf = RandomForestClassifier(n_estimators=100)

# Train the model on the training set
rf.fit(X_train, y_train)

# Use the model to make predictions on the testing set
y_pred = rf.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Get the decision rules for every tree in the forest

# for i, tree in enumerate(rf.estimators_):
#     print(f"Tree {i + 1}")
#     print(export_text(tree, feature_names=dataset.feature_names))

# print the outcome of each test case with the prediction
for i, test_case in enumerate(X_test):
    print(f"Test case {i + 1}")
    print(f"Lexical score: {test_case[0]}")
    print(f"Instance overlap: {test_case[1]}")
    print(f"Subclass overlap: {test_case[2]}")
    print(f"Description overlap: {test_case[3]}")
    print(f"Prediction: {y_pred[i]}")
    print(f"Actual: {y_test[i]}")
    print()
