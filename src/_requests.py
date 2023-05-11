import requests
from _types import WikiDataSearchEntitiesResponse
import re

from util2 import (
    JsonUpdater,
    PickleUpdater,
    parse_entity_description,
    parse_entity_properties,
    parse_entity_statements,
    parse_entity_title,
    pickle_save,
    progress,
)

API_URL = "https://www.wikidata.org/w/api.php"

ENTITY_RE = r"^Q\d+$"


# Request limit exception
class RateLimitException(Exception):
    pass


fetch_entity_updater = PickleUpdater("/datasets/wikidata_fetch_entity_cache_2023", save_interval=1200)
entity_query_updater = PickleUpdater("/datasets/wikidata_entity_query_cache_2023", save_interval=0)
entity_search_updater = PickleUpdater("/datasets/wikidata_entity_search_cache_2023", save_interval=0)


def wikidata_fetch_entities(ids: list[int], lang: str = "en", chunk_size: int = 50):
    if not fetch_entity_updater.data_loaded:
        fetch_entity_updater.load_data()

    # Remove queries that are already in the cache
    ids = [id for id in ids if id not in fetch_entity_updater.data]
    if len(ids) == 0:
        return

    # Split queries into chunks of 50
    query_chunks = [ids[i : i + chunk_size] for i in range(0, len(ids), chunk_size)]

    with progress:
        for chunk in progress.track(query_chunks, description="Fetching entities"):
            params = {
                "action": "wbgetentities",
                "languages": lang,
                "format": "json",
                "ids": "|".join([f"Q{q}" for q in chunk]),
            }

            data = requests.get(API_URL, params=params)
            data_json = data.json()

            assert data.status_code == 200, f"Request failed: {data_json}"
            assert data_json["success"] == 1, f"Request failed: {data_json}"

            for entity_id in chunk:
                entity_data = data_json["entities"][f"Q{entity_id}"]
                title = parse_entity_title(entity_data) or ""
                description = parse_entity_description(entity_data) or ""
                statements = []
                if "claims" in entity_data:
                    for claims in entity_data["claims"].values():
                        for claim in claims:
                            if claim["mainsnak"]["snaktype"] == "novalue" or claim["mainsnak"]["snaktype"] == "somevalue":
                                continue
                            prop = int(claim["mainsnak"]["property"][1:])
                            type = claim["mainsnak"]["datatype"]
                            value = claim["mainsnak"]["datavalue"]["value"]
                            statements.append((prop, type, value))
                fetch_entity_updater.update_data(entity_id, (title, description, statements))

    fetch_entity_updater.close_data()


def get_entity(entity_id: int):
    if not fetch_entity_updater.data_loaded:
        fetch_entity_updater.load_data()

    assert entity_id in fetch_entity_updater.data, f"Entity {entity_id} not found in cache."

    return fetch_entity_updater.data[entity_id]


# OLDLDLD
# -------------
# -------------
# -------------

# entity_query_updater = JsonUpdater("/datasets/wikidata_entity_query_cache.json")
# entity_search_updater = JsonUpdater("/datasets/wikidata_entity_search_cache.json")
get_entity_updater = JsonUpdater("/datasets/wikidata_get_entity_cache.json")
get_property_updater = JsonUpdater("/datasets/wikidata_get_property_cache.json")


def wikidata_entity_query(query: str) -> list[str]:
    if not entity_query_updater.data_loaded:
        entity_query_updater.load_data()

    if query in entity_query_updater.data:
        entities = entity_query_updater.data[query]
        if not all([bool(re.match(ENTITY_RE, e)) for e in entities]):
            print(f"Found invalid entity ID in cache: {entities}... Removing from cache.")
            entity_query_updater.delete_data(query)
        else:
            return entity_query_updater.data[query]

    params = {
        "action": "query",
        "srsearch": query,
        "list": "search",
        "format": "json",
    }

    try:
        data = requests.get(API_URL, params=params)
        if data.status_code == 429:
            raise RateLimitException()

        res = data.json()
        if not "query" in res or not "search" in res["query"]:
            pickle_save({"query": query, "data": data})
            return []

        entity_ids = [result["title"] for result in res["query"]["search"]]
        entity_ids = [e for e in entity_ids if bool(re.match(ENTITY_RE, e))]

        entity_query_updater.update_data(query, entity_ids)

        return entity_ids
    except Exception as e:
        pickle_save({"e": e, "query": query, "data": data})
        return []


def wikidata_entity_search(query: str, limit: int = 30, lang: str = "en") -> list[str]:
    """
    Fetches a list of entities matching the given query from the Wikidata API.

    Parameters
    ----------
    mention : str
        The mention to search for.
    limit : int, optional
        The maximum number of candidates to return, by default 10
    lang : str, optional
        The language to search in, by default "en"

    Returns
    -------
    list[str]
        A list of entity IDs.

    Example
    -------
    >>> get_candidates("Barack Obama", limit=3)
    ['Q76', 'Q47513588', 'Q59661289']
    """

    if not entity_search_updater.data_loaded:
        entity_search_updater.load_data()

    other_ids = wikidata_entity_query(query)

    if query in entity_search_updater.data:
        entities = entity_query_updater.data[query]
        if not all([bool(re.match(ENTITY_RE, e)) for e in entities]):
            print(f"Found invalid entity ID in cache: {entities}... Removing from cache.")
            entity_search_updater.delete_data(query)
        else:
            return list(set(other_ids + entity_search_updater.data[query]))

    params = {
        "action": "wbsearchentities",
        "language": lang,
        "format": "json",
        "search": query,
        "limit": f"{limit}",
    }
    data = requests.get(API_URL, params=params)
    if data.status_code == 429:
        raise RateLimitException()

    res = data.json()
    if "search-continue" in res:
        res["search_continue"] = res.pop("search-continue")
    res: WikiDataSearchEntitiesResponse = res

    search_results = res["search"]
    entity_ids = [result["id"] for result in search_results]
    entity_ids = [e for e in entity_ids if bool(re.match(ENTITY_RE, e))]

    entity_search_updater.update_data(query, entity_ids)

    return list(set(other_ids + entity_ids))


def wikidata_get_entities(entity_ids: list[int], lang: str = "en") -> list[dict]:
    if not get_entity_updater.data_loaded:
        get_entity_updater.load_data()

    res = []

    for entity_id in entity_ids:
        if f"{entity_id}" in get_entity_updater.data:
            res.append({f"{entity_id}": get_entity_updater.data[f"{entity_id}"]})
            entity_ids.remove(entity_id)
            continue

    params = {
        "action": "wbgetentities",
        "languages": lang,
        "format": "json",
        "ids": "|".join([f"Q{entity_id}" for entity_id in entity_ids]),
    }

    data = requests.get(API_URL, params=params)
    if data.status_code == 429:
        raise RateLimitException()

    for entity_id in entity_ids:
        entity = data.json()["entities"][f"Q{entity_id}"]

        entity_data = {
            "title": parse_entity_title(entity) or "",
            "description": parse_entity_description(entity) or "",
            "properties": parse_entity_properties(entity),
        }

        get_entity_updater.update_data(entity_id, entity_data)
        res.append({f"{entity_id}": entity_data})

    return res


def wikidata_get_properties(property_ids: list[str], lang: str = "en") -> list[dict]:
    if not get_property_updater.data_loaded:
        get_property_updater.load_data()

    res = []

    for property_id in property_ids:
        if f"{property_id}" in get_property_updater.data:
            res.append({f"{property_id}": get_property_updater.data[f"{property_id}"]})
            property_ids.remove(property_id)
            continue

    params = {
        "action": "wbgetentities",
        "languages": lang,
        "format": "json",
        "ids": "|".join([f"{property_id}" for property_id in property_ids]),
    }

    data = requests.get(API_URL, params=params)
    if data.status_code == 429:
        raise RateLimitException()

    for property_id in property_ids:
        json = data.json()
        property = data.json()["entities"][f"{property_id}"]

        if not "labels" in property:
            continue

        if not "en" in property["labels"]:
            continue

        if not "value" in property["labels"]["en"]:
            continue

        property_data = {
            "label": property["labels"]["en"]["value"],
        }

        x = 1

        get_property_updater.update_data(property_id, property_data)
        res.append({f"{property_id}": property_data})

    return res


def wikidata_get_entity(entity_id: int, lang: str = "en") -> dict:
    """
    Fetches an entity from the Wikidata API.

    Parameters
    ----------
    entity_id : int
        The ID of the entity to fetch.
    lang : str, optional
        The language to fetch the entity in, by default "en"

    Returns
    -------
    dict
        The entity.

    Example
    -------
    >>> wikidata_get_entity(76)
    """
    if not get_entity_updater.data_loaded:
        get_entity_updater.load_data()

    if f"{entity_id}" in get_entity_updater.data:
        return get_entity_updater.data[f"{entity_id}"]

    params = {
        "action": "wbgetentities",
        "languages": lang,
        "format": "json",
        "ids": f"Q{entity_id}",
    }

    data = requests.get(API_URL, params=params)
    if data.status_code == 429:
        raise RateLimitException()

    entity = data.json()["entities"][f"Q{entity_id}"]

    entity_data = {
        "title": parse_entity_title(entity) or "",
        "description": parse_entity_description(entity) or "",
        "properties": parse_entity_properties(entity),
    }

    get_entity_updater.update_data(entity_id, entity_data)

    return entity_data


def fetch_property_labels():
    from util import progress

    MAX_CACHE_SIZE = 50
    current_cache = set()
    cache_history = set()

    with progress:
        for entity in progress.track(get_entity_updater.data.values(), description="Fetching property labels"):
            for p, _ in entity["properties"]:
                if p in cache_history or p in current_cache:
                    continue

                current_cache.add(p)
                cache_history.add(p)

                if len(current_cache) >= MAX_CACHE_SIZE:
                    wikidata_get_properties(list(current_cache))
                    current_cache = set()


# fetch_property_labels()
