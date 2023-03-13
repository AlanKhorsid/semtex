import requests
from _types import WikiDataSearchEntitiesResponse, validate_wikidata_search_entities_response
from util import pickle_save

API_URL = "https://www.wikidata.org/w/api.php"

# Request limit exception
class RateLimitException(Exception):
    pass


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

    # try:
    #     validate_wikidata_search_entities_response(res)
    # except Exception as e:
    #     print('Error validating wikidata search entities response!!!!')
    #     print(e)
    #     pickle_save(res)

    search_results = res["search"]
    entity_ids = [result["id"] for result in search_results]

    return entity_ids


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

    params = {
        "action": "wbgetentities",
        "languages": lang,
        "format": "json",
        "ids": f"Q{entity_id}",
    }

    data = requests.get(API_URL, params=params)
    if data.status_code == 429:
        raise RateLimitException()

    return data.json()["entities"][f"Q{entity_id}"]
