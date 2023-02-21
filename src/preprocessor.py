#import pandas
#import numpy
from autocorrect import Speller
from spellchecker import SpellChecker

def spellcheck(inputString: str) -> {str}:
    spell = SpellChecker()
    candidates = []
    misspelled = spell.unknown(inputString.split(" "))

    for word in misspelled:
        candidates.append(spell.candidates(word))

    return candidates

def autocorrect(inputString: str) -> str:
    spell = Speller()
    autocorrectedString = spell(inputString)

    return autocorrectedString

string = "adm"

x = spellcheck(string)
for cand in x:
    print(cand)
print(autocorrect(string))
