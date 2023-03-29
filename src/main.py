from functools import reduce
import itertools
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
import gensim.downloader as api
from gensim.models.doc2vec import Doc2Vec
import torch
from transformers import RobertaTokenizer, RobertaModel
import numpy as np

from _requests import wikidata_get_entities

i = 0
PICKLE_FILE_NAME = "test-2022-bing"

# ----- Open dataset -----
# cols = open_dataset(dataset="validation", disable_spellcheck=False)
# pickle_save(cols, f"{PICKLE_FILE_NAME}-{i}")
# i = i + 1 if i < 9 else 1
cols: list[Column] = pickle_load(f"{PICKLE_FILE_NAME}", is_dump=True)

# model = api.load("conceptnet-numberbatch-17-06-300")
# pickle_save(model, "conceptnet-numberbatch-17-06-300-model")
# model = pickle_load("conceptnet-numberbatch-17-06-300-model", is_dump=True)


# x = model.wmdistance(
#     "Translations of One Thousand and One Nights, list of translations of literary work",
#     "One Thousand and One Nights, 1990 studio album by Hacken Lee",
# )

# print(x)

# x = []


# import gensim
# import gensim.downloader as api

# dataset = api.load("text8")
# data = [d for d in dataset]


# def tagged_document(list_of_list_of_words):
#     for i, list_of_words in enumerate(list_of_list_of_words):
#         yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])


# data_for_training = list(tagged_document(data))
# print(data_for_training[:1])
# model = gensim.models.doc2vec.Doc2Vec(vector_size=40, min_count=2, epochs=30)
# model.build_vocab(data_for_training)
# model.train(data_for_training, total_examples=model.corpus_count, epochs=model.epochs)
# pickle_save(model, "text8-model")


# model = pickle_load("text8-model", is_dump=True)
# print(model.infer_vector(["violent", "means", "to", "destroy", "the", "organization"]))

# y = model.similarity_unseen_docs(["zirconium dioxide - chemical compound"], ["zirconium-99 - isotope of zirconium"])
# z = model.similarity_unseen_docs(
#     "zirconium dioxide - chemical compound".split(), "zirconium-99 - isotope of zirconium".split()
# )
# a = model.similarity_unseen_docs(
#     "zirconium dioxide chemical compound".split(), "zirconium-99 isotope of zirconium".split()
# )

model_name = "roberta-large"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaModel.from_pretrained(model_name)


def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].detach().numpy()


# def similarity(a, b, model):
#     return model.similarity_unseen_docs(a.split(), b.split())

num_correct = 0
num_total = 0

for col in cols:
    total_combinations = reduce(lambda x, y: x * y, [len(c.candidates) for c in col.cells])
    if total_combinations > 1000000 or total_combinations == 0:
        print(f"Skipping because it has {total_combinations} combinations")
        continue
    print(f"Processing with {total_combinations} combinations")
    print("")

    embeddings = {
        candidate.id: get_embedding(candidate.to_sentence) for cell in col.cells for candidate in cell.candidates
    }

    similarities = {}
    lists = []
    with progress:
        # for i, cell in progress.track(enumerate(col.cells), total=len(col.cells), description="Building lists"):
        for i, cell in enumerate(col.cells):
            cands = []
            for candidate in cell.candidates:
                cands.append(candidate)
            lists.append(cands)

            for j, cell in enumerate(col.cells):
                if i == j:
                    continue
                for cand1 in col.cells[i].candidates:
                    for cand2 in col.cells[j].candidates:
                        if (cand1.id, cand2.id) in similarities or (cand2.id, cand1.id) in similarities:
                            continue
                        similarities[(cand1.id, cand2.id)] = torch.cosine_similarity(
                            torch.tensor(embeddings[cand1.id]), torch.tensor(embeddings[cand2.id])
                        )

    best_avg_similarity = 0
    best_combination = None

    with progress:
        for cand in itertools.product(*lists):
            sims = [
                similarities[(x.id, y.id)] if (x.id, y.id) in similarities else similarities[(y.id, x.id)]
                for x, y in itertools.combinations(cand, 2)
            ]
            avg_similarity = np.mean([s.numpy() for s in sims])
            if avg_similarity > best_avg_similarity:
                best_avg_similarity = avg_similarity
                best_combination = cand
                print(f"New best avg similarity: {best_avg_similarity}")
                print("---------------------------")
                for i, c in enumerate(best_combination):
                    print(f"{'CORRECT ' if c.id == col.cells[i].correct_candidate.id else '        '}{c.to_sentence}")
                print("---------------------------")
                print("")

    for i, cand in enumerate(best_combination):
        if cand.id == col.cells[i].correct_candidate.id:
            num_correct = num_correct + 1
        num_total = num_total + 1

    print(f"Accuracy: {num_correct / num_total}")
    print("")
    print("")
    print("")

    # min_dist, selections = min_total_distance(model, lists)
    # print(min_dist, selections)
    # x = 1
