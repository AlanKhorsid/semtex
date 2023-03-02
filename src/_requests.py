import requests

API_URL = "https://www.wikidata.org/w/api.php"


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

    search_results = data.json()["search"]
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
    return data.json()["entities"][f"Q{entity_id}"]