from flair.data import Sentence
from flair.models import SequenceTagger

sentence = Sentence(["George Washington"])

tagger = SequenceTagger.load("ner")
tagger.predict(sentence)

for entity in sentence.get_spans("ner"):
    print(entity.tag, entity.score)
