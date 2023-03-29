import gensim.downloader as api
import numpy as np

# Load the ConceptNet Numberbatch embeddings
model = api.load("conceptnet-numberbatch-17-06-300")


# Define a function to find the cosine similarity between two vectors
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


word1 = "cat"
word2 = "dog"

# Get the embeddings for the words
embedding1 = model[word1]
embedding2 = model[word2]

# Calculate the cosine similarity between the embeddings
similarity = cosine_similarity(embedding1, embedding2)

print(f"The similarity between '{word1}' and '{word2}' is {similarity:.4f}")
