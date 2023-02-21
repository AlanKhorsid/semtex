#import pandas
#import numpy
from autocorrect import Speller

def autocorrect(inputString: str) -> str:
    spell = Speller()
    autocorrectedString = spell(inputString)

    return autocorrectedString

print(autocorrect("barack obma"))
