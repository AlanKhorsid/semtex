import random
from tqdm import tqdm
from classes import CandidateSet, Column
from util import (
    cluster_data,
    ensemble_catboost_regression,
    ensemble_hist_gradient_boost_regression,
    ensemble_xgboost_regression,
    ensemble_gradient_boost_regression,
    evaluate_model,
    open_dataset,
    random_forest_regression,
    pickle_save,
    pickle_load,
)
from sklearn.model_selection import ParameterGrid, ParameterSampler
import random
from transformers import AutoTokenizer, AutoModel, RobertaTokenizer, RobertaModel

i = 0
PICKLE_FILE_NAME = "test-2022-bing"

# ----- Open dataset -----
print("Opening dataset...")
# cols = open_dataset(dataset="validation", disable_spellcheck=False)
# pickle_save(cols, f"{PICKLE_FILE_NAME}-{i}")
# i = i + 1 if i < 9 else 1
cols: list[Column] = pickle_load(f"{PICKLE_FILE_NAME}", is_dump=True)
# cols2: list[Column] = pickle_load("test-2022-bing", is_dump=True)

# ----- Fetch candidates -----
print("Fetching candidates...")
while not all([col.all_cells_fetched for col in cols]):
    for col in tqdm(cols):
        if col.all_cells_fetched:
            continue
        col.fetch_cells()
        pickle_save(cols, f"{PICKLE_FILE_NAME}-{i}")
        i = i + 1 if i < 9 else 1


model_name = "bert-large-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
# model_name = "roberta-large"
# tokenizer = RobertaTokenizer.from_pretrained(model_name)
# model = RobertaModel.from_pretrained(model_name)


def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].detach().numpy()


print("Generating sentences...")
num_correct = 0
num_total = 0
for col in cols:
    predictions = col.predict_nlp(get_embedding)

    for i, prediction in enumerate(predictions):
        num_total += 1
        if prediction is None:
            continue
        if prediction.id == col.cells[i].correct_id:
            num_correct += 1

    print(f"Accuracy: {num_correct / num_total}")
