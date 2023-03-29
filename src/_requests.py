import requests
from _types import WikiDataSearchEntitiesResponse, validate_wikidata_search_entities_response
from util import (
    parse_entity_description,
    parse_entity_properties,
    parse_entity_title,
    pickle_save,
    pickle_load,
    JsonUpdater,
)
import threading
import re

API_URL = "https://www.wikidata.org/w/api.php"

ENTITY_RE = r"^Q\d+$"

# Request limit exception
class RateLimitException(Exception):
    pass


entity_query_updater = JsonUpdater("/datasets/wikidata_entity_query_cache.json")
entity_search_updater = JsonUpdater("/datasets/wikidata_entity_search_cache.json")
get_entity_updater = JsonUpdater("/datasets/wikidata_get_entity_cache.json")
get_property_updater = JsonUpdater("/datasets/wikidata_get_property_cache.json")


def wikidata_entity_query(query: str) -> list[str]:
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


def wikidata_get_property(property_id: str, lang: str = "en") -> list[dict]:
    if property_id in get_property_updater.data:
        return get_property_updater.data[property_id]

    raise NotImplementedError()

    params = {
        "action": "wbgetentities",
        "languages": lang,
        "format": "json",
        "ids": f"{property_id}",
    }

    data = requests.get(API_URL, params=params)
    if data.status_code == 429:
        raise RateLimitException()

    property = data.json()["entities"][f"{property_id}"]

    if not "labels" in property:
        raise ValueError(f"Property {property_id} does not have a label.")

    if not "en" in property["labels"]:
        raise ValueError(f"Property {property_id} does not have an English label.")

    if not "value" in property["labels"]["en"]:
        raise ValueError(f"Property {property_id} does not have an English label value.")

    property_data = {
        "label": property["labels"]["en"]["value"],
    }

    get_property_updater.update_data(property_id, property_data)

    return property_data
