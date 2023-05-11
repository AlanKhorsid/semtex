from util2 import PickleUpdater, pickle_load, pickle_save


fetch_entity_updater = pickle_load("wikidata_fetch_entity_cache_2023")
updated = {}

for k, entity in fetch_entity_updater.items():
    title, description, statements = entity
    new_statements = []

    for prop, type, value in statements:
        if type == "wikibase-item" or type == "quantity" or type == "time" or type == "monolingualtext":
            new_statements.append((prop, type, value))
    
    updated[k] = (title, description, new_statements)

        


pickle_save(updated, "wikidata_fetch_entity_cache_2023_stripped")