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

API_URL = "https://www.wikidata.org/w/api.php"

# Request limit exception
class RateLimitException(Exception):
    pass


entity_query_updater = JsonUpdater("/datasets/wikidata_entity_query_cache.json")
entity_search_updater = JsonUpdater("/datasets/wikidata_entity_search_cache.json")
get_entity_updater = JsonUpdater("/datasets/wikidata_get_entity_cache.json")


def wikidata_entity_query(query: str) -> list[str]:
    if query in entity_query_updater.data:
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
