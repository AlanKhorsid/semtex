from collections import Counter
import requests


def get_desc(entity_name):
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "language": "en",
        "format": "json",
        "search": entity_name
    }
    try:
        data = requests.get(url, params=params)
        entity = data.json()["search"][0]
        return entity["description"].split()
    except:
        return None


def find_most_common_word(entity_types):
    function_words = ["the", "of", "a", "an", "in", "to", "it"]
    all_words = [word.lower(
    ) for entity_type in entity_types for word in entity_type if word.lower() not in function_words]
    word_count = {}
    for word in all_words:
        word_count[word] = word_count.get(word, 0) + 1
    most_common_word = max(word_count, key=word_count.get)
    return most_common_word


def find_second_most_common_word(entity_types):
    function_words = {"the", "of", "a", "an", "in", "to", "it"}
    all_words = [word.lower(
    ) for entity_type in entity_types for word in entity_type if word.lower() not in function_words]
    word_count = Counter(all_words)
    sorted_word_count = sorted(
        word_count.items(), key=lambda x: x[1], reverse=True)
    return sorted_word_count[1][0] if len(sorted_word_count) > 1 else None


def main(entities):
    entity_types = [get_desc(entity_name) for entity_name in entities]
    most_common_word = find_most_common_word(entity_types)
    second_most_common_word = find_second_most_common_word(entity_types)
    print("The most common word among the entities is:", most_common_word)
    print("The second most common word among the entities is:",
          second_most_common_word)


entities = [input("Enter name of entity {}: ".format(i + 1)) for i in range(5)]

main(entities)
