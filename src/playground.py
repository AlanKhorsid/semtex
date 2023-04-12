from util import pickle_load, pickle_save

# x1 = pickle_load("all-test-tag", is_dump=True)

# # get tags and tag ratio
# for col in x1:
#     print(f"Column: {col.cells}")
#     for cell in col.cells:
#         print(f"Cell: {cell}")
#         print()
#         for candidate in cell.candidates:
#             print(f"Candidate: {candidate.to_sentence}")
#             print(f"Tag:    {candidate.tag}")
#             print(f"Tag ratio:    {candidate.tag_ratio}")
#             print()

# x2 = pickle_load("best-model-so-far", is_dump=True)

# print(x2.get_params())
# print(x2)
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


# Define the function to find the most semantically similar sentence from a list
def find_most_similar(target_sentence, sentence_list):
    # Tokenize the target sentence into words
    target_words = word_tokenize(target_sentence)

    # Remove stop words from the target sentence
    stop_words = set(stopwords.words("english"))
    target_words = [word for word in target_words if not word.lower() in stop_words]

    # Lemmatize the words in the target sentence
    lemmatizer = WordNetLemmatizer()
    target_words = [lemmatizer.lemmatize(word) for word in target_words]

    # Create synsets for the words in the target sentence
    target_synsets = [wn.synsets(word) for word in target_words]
    target_synsets = [synset for sublist in target_synsets for synset in sublist]

    # Loop through each sentence in the list and find the maximum similarity score
    max_similarity = 0
    most_similar_sentence = ""
    for sentence in sentence_list:
        # Tokenize the sentence into words
        words = word_tokenize(sentence)

        # Remove stop words from the sentence
        words = [word for word in words if not word.lower() in stop_words]

        # Lemmatize the words in the sentence
        words = [lemmatizer.lemmatize(word) for word in words]

        # Create synsets for the words in the sentence
        synsets = [wn.synsets(word) for word in words]
        synsets = [synset for sublist in synsets for synset in sublist]

        # Find the maximum similarity score between synsets from the target sentence and the current sentence
        similarity_sum = 0
        for target_synset in target_synsets:
            for synset in synsets:
                similarity = target_synset.path_similarity(synset)
                if similarity is not None:
                    similarity_sum += similarity

        similarity_score = similarity_sum / (len(target_synsets) * len(synsets))

        # Update the maximum similarity score and most similar sentence
        if similarity_score > max_similarity:
            max_similarity = similarity_score
            most_similar_sentence = sentence

    return most_similar_sentence, max_similarity


# Example usage
target_sentence = "Islamic Emirate of Afghanistan is a Taliban-led partially recognized government of Afghanistan from 1996 to 2001 and is an instance of sovereign state, emirate, historical country, state with limited recognition"
sentence_list = [
    "Transnistria is a de facto unrecognized state in Eastern Europe that has declared independence from Moldova and is an instance of state with limited recognition, country, landlocked country, unitary state, Rechtsstaat, social state, secular state",
    "Parliament of Transnistria is a Legislature of Transnistria and is an instance of unicameral legislature",
    "Transnistria is an instance of geographical region, disputed territory",
    "coat of arms of Transnistria is a national coat of arms of Transnistria and is an instance of national coat of arms",
]
most_similar_sentence, similarity_score = find_most_similar(
    target_sentence, sentence_list
)
print("The most semantically similar sentence is:", most_similar_sentence)
print("The similarity score is:", similarity_score)
# print all scores
for sentence in sentence_list:
    most_similar_sentence, similarity_score = find_most_similar(
        target_sentence, [sentence]
    )
    print(f"Similarity score for {sentence} is {similarity_score}")
