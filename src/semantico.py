import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Load the pre-trained BERT model and tokenizer
model_name = "bert-large-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


def get_best_candidates_TEST(groups):
    # Combine all groups into a single list
    all_names = [name for group in groups for name in group]

    # Define a function to get embeddings from the model
    def get_embedding(text):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].detach().numpy()

    # Calculate embeddings for each name
    embeddings = [get_embedding(name) for name in all_names]

    # Construct the similarity matrix
    similarity_matrix = np.zeros((len(all_names), len(all_names)))
    for i, name1 in enumerate(all_names):
        for j, name2 in enumerate(all_names):
            if any(name1 in group and name2 in group for group in groups):
                continue  # Skip comparing strings within the same group
            emb1 = embeddings[i]
            emb2 = embeddings[j]
            # Cosine similarity
            similarity_matrix[i][j] = torch.cosine_similarity(
                torch.tensor(emb1), torch.tensor(emb2)
            )

    # Get the best candidate from each group
    best_candidates = []
    for i, group in enumerate(groups):
        # print(f"\nGroup {i + 1}:")
        max_similarity = -1
        best_candidate = None
        for name in group:
            # Calculate the average similarity of the name to all other names outside the group
            indices = [j for j in range(len(all_names)) if all_names[j] not in group]
            similarities = similarity_matrix[all_names.index(name), indices]
            avg_similarity = np.mean(similarities)
            # print(f"{name} - {avg_similarity:.2f}")
            # Update the best candidate if this name has a higher average similarity
            if avg_similarity > max_similarity:
                max_similarity = avg_similarity
                best_candidate = name
        best_candidates.append(best_candidate)
        print(f"Best candidate: {best_candidate}")

    return best_candidates


group1 = [
    "Barack Obama, is a president of the United States from 2009 to 2017 and an instance of a human",
    "Obama, is a city in Fukui prefecture, Japan and an instance of city of Japan",
    "Obama, is a genus of worms and an instance of taxon",
    "Obama, is a family name and an instance of family name",
]

group2 = [
    "Vladimir Putin, is a President of Russia (1999-2008, 2012-present) and an instance of human",
    "Vladimir Putin, is a duplicate article and a instance of Wikimedia permanent duplicate item",
    "Vladimir Putin, is a 2018 Russia presidential campaign of Vladimir Putin and a instance of presidential campaign, of, Vladimir Putin",
    "Russia under Vladimir Putin, is a overview of the Presidency of Vladimir Putin and a instance of historical period",
]

group3 = [
    "Emmanuel Macron, is a President of France since 2017 and a instance of human",
    "Emmanuel Macron, is a les coulisses d'une victoire, 2017 film and a instance of film",
    "protests against Emmanuel Macron, is a series of protests in France and a instance of protest of presidency of Emmanuel Macron",
    "Revolution, is a book by Emmanuel Macron and a instance of written work",
]

group4 = [
    "Xi Jinping, is a General Secretary of the Chinese Communist Party and an instance of human",
    "Xi Jinping Thought on Socialism with Chinese Characteristics for a New Era, is a political theory attributed to Xi Jinping, the General Secretary of the Chinese Communist Party and an instance of dictatorship, authoritarianism, totalitarianism",
    "Xi Jinping Administration, is a the CCP Central Committee with Comrade Xi Jinping at its core and an instance of political theory, political ideology",
    "Xi Jinping, is a Chinese administration headed by Communist Party General Secretary Xi Jinping and an instance of dictatorship, Leadership of the People's Republic of China",
]

group5 = [
    "Olaf Scholz, is a German politician (SPD) and 9th Federal Chancellor of Germany (since 2021) and an instance of human",
    "Olaf Scholz, is a German ice hockey player and an instance of human",
    "Scholz cabinet, is a  cabinet in the German federal government headed by Chancellor Scholz and an instance of Cabinet of the Federal Republic of Germany",
    "Christel Scholz is a instance of human",
]

get_best_candidates_TEST([group1, group2, group3, group4, group5])
