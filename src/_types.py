from enum import Enum
from typing import Any, Dict, TypedDict, Literal, Union


class ClaimType(Enum):
    STRING = 1
    TIME = 2
    ENTITY = 3
    QUANTITY = 4
    LEXEME = 5
    PROPERTY = 6
    # GLOBE_COORDINATES = 5
    # MONOLINGUAL_TEXT = 6
    # MULTILINGUAL_TEXT = 7
    # URL = 8
    # EXTERNAL_ID = 9
    UNKNOWN = 10


Entity = TypedDict("Entity", {"id": str, "claims": dict["Entity"]})

Claim = TypedDict("Claim", {"type": ClaimType, "value": str})

DiscoveredEntity = TypedDict("DiscoveredEntity", {"candidate": str, "index": int})

class SearchInfo(TypedDict):
    search: str


class Label(TypedDict):
    value: str
    language: str


class Description(TypedDict):
    value: str
    language: str


class Display(TypedDict):
    label: Label
    description: Union[Description, None]


class Match(TypedDict):
    type: Literal["label", "alias"]
    language: str
    text: str


class Search(TypedDict):
    id: str
    title: str
    pageid: int
    display: Display
    repository: Literal["wikidata"]
    url: str
    concepturi: str
    label: str
    description: Union[str, None]
    match: Match
    aliases: Union[list[str], None]


class WikiDataSearchEntitiesResponse(TypedDict):
    searchinfo: SearchInfo
    search: list[Search]
    search_continue: Union[int, None]
    success: Literal[1]


# FOR TYPE CHECKING ONLY


def validate_search_info(search_info: Dict[str, Any]) -> None:
    if not isinstance(search_info, dict):
        raise TypeError(f"Invalid type for search_info: {type(search_info)}")
    
    # search
    if "search" not in search_info or not isinstance(search_info["search"], str):
        raise ValueError(f"Invalid search_info value: {search_info}")

    for key in search_info:
        if key not in ["search"]:
            raise ValueError(f"Invalid key in search_info: {key}")


def validate_label(label: Dict[str, Any]) -> None:
    if not isinstance(label, dict):
        raise TypeError(f"Invalid type for label: {type(label)}")
    
    # value
    if "value" not in label or not isinstance(label["value"], str):
        raise ValueError(f"Invalid label value: {label}")
    
    # language
    if "language" not in label or not isinstance(label["language"], str):
        raise ValueError(f"Invalid label language: {label}")

    for key in label:
        if key not in ["value", "language"]:
            raise ValueError(f"Invalid key in label: {key}")


def validate_description(description: Dict[str, Any]) -> None:
    if not isinstance(description, dict):
        raise TypeError(f"Invalid type for description: {type(description)}")
    
    # value
    if "value" not in description or not isinstance(description["value"], str):
        raise ValueError(f"Invalid description value: {description}")
    
    # language
    if "language" not in description or not isinstance(description["language"], str):
        raise ValueError(f"Invalid description language: {description}")
    
    for key in description:
        if key not in ["value", "language"]:
            raise ValueError(f"Invalid key in description: {key}")


def validate_display(display: Dict[str, Any]) -> None:
    if not isinstance(display, dict):
        raise TypeError(f"Invalid type for display: {type(display)}")
    
    # label
    if "label" not in display:
        raise ValueError(f"Invalid display label: {display}")
    validate_label(display["label"])

    # description
    if "description" in display:
        validate_description(display["description"])
    
    for key in display:
        if key not in ["label", "description"]:
            raise ValueError(f"Invalid key in display: {key}")


def validate_match(match: Dict[str, Any]) -> None:
    if not isinstance(match, dict):
        raise TypeError(f"Invalid type for match: {type(match)}")

    # type
    if "type" not in match or match["type"] not in ["label", "alias"]:
        raise ValueError(f"Invalid match type: {match}")
    
    # language
    if "language" not in match or not isinstance(match["language"], str):
        raise ValueError(f"Invalid match language: {match}")
    
    # text
    if "text" not in match or not isinstance(match["text"], str):
        raise ValueError(f"Invalid match text: {match}")

    for key in match:
        if key not in ["type", "language", "text"]:
            raise ValueError(f"Invalid key in match: {key}")


def validate_search(search: Dict[str, Any]) -> None:
    if not isinstance(search, dict):
        raise TypeError(f"Invalid type for search: {type(search)}")
    
    # id
    if "id" not in search or not isinstance(search["id"], str):
        raise ValueError(f"Invalid search id: {search}")
    
    # title
    if "title" not in search or not isinstance(search["title"], str):
        raise ValueError(f"Invalid search title: {search}")
    
    # pageid
    if "pageid" not in search or not isinstance(search["pageid"], int):
        raise ValueError(f"Invalid search pageid: {search}")
    
    # display
    if "display" not in search:
        raise ValueError(f"Invalid search display: {search}")
    validate_display(search.get("display", {}))

    # repository
    if "repository" not in search or search["repository"] != "wikidata":
        raise ValueError(f"Invalid search repository: {search}")
    
    # url
    if "url" not in search or not isinstance(search["url"], str):
        raise ValueError(f"Invalid search url: {search}")
    
    # concepturi
    if "concepturi" not in search or not isinstance(search["concepturi"], str):
        raise ValueError(f"Invalid search concepturi: {search}")
    
    # label
    if "label" not in search or not isinstance(search["label"], str):
        raise ValueError(f"Invalid search label: {search}")
    
    # description
    if "description" in search and not isinstance(search["description"], str):
        raise ValueError(f"Invalid search description: {search}")
    
    # match
    if "match" not in search:
        raise ValueError(f"Invalid search match: {search}")
    validate_match(search.get("match", {}))

    # aliases
    if "aliases" in search and not isinstance(search["aliases"], list):
        raise ValueError(f"Invalid search aliases: {search}")
    if "aliases" in search:
        for alias in search["aliases"]:
            if not isinstance(alias, str):
                raise ValueError(f"Invalid search alias: {alias}")

    for key in search:
        if key not in ["id", "title", "pageid", "display", "repository", "url", "concepturi", "label", "description", "match", "aliases"]:
            raise ValueError(f"Invalid key in search: {key}")


def validate_wikidata_search_entities_response(wikidata_search_entities_response: Dict[str, Any]) -> None:
    if not isinstance(wikidata_search_entities_response, dict):
        raise TypeError(f"Invalid type for wikidata_search_entities_response: {type(wikidata_search_entities_response)}")
    
    # searchinfo
    if "searchinfo" not in wikidata_search_entities_response:
        raise ValueError(f"Invalid wikidata_search_entities_response searchinfo: {wikidata_search_entities_response}")
    validate_search_info(wikidata_search_entities_response.get("searchinfo", {}))

    # search
    if "search" not in wikidata_search_entities_response or not isinstance(wikidata_search_entities_response["search"], list):
        raise ValueError(f"Invalid wikidata_search_entities_response search: {wikidata_search_entities_response}")
    for search in wikidata_search_entities_response["search"]:
        validate_search(search)

    # search_continue
    if "search_continue" in wikidata_search_entities_response and not isinstance(wikidata_search_entities_response["search_continue"], int):
        raise ValueError(f"Invalid wikidata_search_entities_response search_continue: {wikidata_search_entities_response}")
    
    # success
    if "success" not in wikidata_search_entities_response or wikidata_search_entities_response["success"] != 1:
        raise ValueError(f"Invalid wikidata_search_entities_response success: {wikidata_search_entities_response}")
    
    for key in wikidata_search_entities_response:
        if key not in ["searchinfo", "search", "search_continue", "success"]:
            raise ValueError(f"Invalid key in wikidata_search_entities_response: {key}")