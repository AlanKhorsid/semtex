from classes import Column
from tqdm import tqdm
from util import (
        ensemble_gradient_boost_regression,
        ensemble_hist_gradient_boost_regression,
        open_dataset,
        pickle_load,
        pickle_save,
        )
import spacy
from collections import Counter
import en_core_web_sm
from enum import Enum
nlp = en_core_web_sm.load()

class SpacyTypes(Enum):
    PERSON = 1
    NORP = 2
    FAC = 3
    ORG = 4
    GPE = 5
    LOC = 6
    PRODUCT = 7
    EVENT = 8
    WORK_OF_ART = 9
    LAW = 10
    LANGUAGE = 11
    DATE = 12
    TIME = 13
    PERCENT = 14
    MONEY = 15
    QUANTITY = 16
    ORDINAL = 17
    CARDINAL = 18

def testspacy(title : str, description : str) -> list[int]:
    labels = []
    results = nlp(f"{title} - {description}")
    for r in results.ents:
        labels.append(r.label_)
        #print((r.text, r.label_))
    labels = list(set(labels))
    labels = [SpacyTypes[label].value for label in labels]
    print(labels)
    return labels

cols: list[Column] = pickle_load("test-data_cols-features")
i = 1
for col in tqdm(cols):
    l1 = []
    l2 = []
    for cell in col.cells:
        for candidate in cell.candidates:
            #print(f"{candidate.title} - {candidate.description}")
            testspacy(candidate.title, candidate.description)
            l1.append(candidate.title)
            l2.append(candidate.description)
        

#testspacy('European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices')

#testspacy('Barack Obama')
#testspacy('barack obama')
