#import pandas
#import numpy
from autocorrect import Speller
from spellchecker import SpellChecker
from itertools import chain, combinations

def spellcheck(inputString: str) -> {str}:
    spell = SpellChecker()
    candidates = []
    misspelled = spell.unknown(inputString.lower().split(" "))

    for word in misspelled:
        candidates.append(spell.candidates(word))

    return candidates

def autocorrect(inputString: str) -> str:
    spell = Speller()
    autocorrectedString = spell(inputString.lower())

    return autocorrectedString

def removeUnwantedCharacters(inputString: str, symbolsToRemove = ["!", "\n", "."]) -> [str]:
    candidateStrings = []
    pSet =  chain.from_iterable(combinations(list(symbolsToRemove), r) for r in range(len(symbolsToRemove)+1))

    for j in pSet:
        cleanedString = inputString
        for k in j:
            cleanedString = cleanedString.replace(k, " ")

        if cleanedString not in candidateStrings:
            candidateStrings.append(cleanedString)

    for i in range(len(candidateStrings)):
        candidateStrings[i] = candidateStrings[i].strip()

    return candidateStrings

# test code
if __name__ == "__main__":
    strings = removeUnwantedCharacters("\"trtle.prsident!obma\"")
    
    print(f"strings: {strings}\n")
    x = []
    for string in strings:
        x.append(spellcheck(string))
    for cand in x:
        print(cand)
