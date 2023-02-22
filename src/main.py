import requests
from enum import Enum
from _types import ClaimType

API_URL = "https://www.wikidata.org/w/api.php"


def get_candidates(mention):
    params = {
        "action": "wbsearchentities",
        "language": "en",
        "format": "json",
        "search": mention,
        "limit": "10",
    }
    try:
        data = requests.get(API_URL, params=params)
        res = [item["id"] for item in data.json()["search"]]
        return [int(item[1:]) for item in res]
    except:
        return []


def parse_claim(claim):
    if claim["mainsnak"]["snaktype"] == "novalue":
        return None

    if claim["mainsnak"]["datatype"] == "wikibase-item":
        return {"type": ClaimType.ENTITY, "value": claim["mainsnak"]["datavalue"]["value"]["id"]}
    else:
        return None

    if claim["mainsnak"]["datatype"] == "wikibase-item":
        return {"type": ClaimType.ENTITY, "value": claim["mainsnak"]["datavalue"]["value"]["id"]}
    elif claim["mainsnak"]["datatype"] == "string":
        return {
            "type": ClaimType.STRING,
            "value": claim["mainsnak"]["datavalue"]["value"],
        }
    elif claim["mainsnak"]["datatype"] == "time":
        return {
            "type": ClaimType.TIME,
            "value": claim["mainsnak"]["datavalue"]["value"]["time"],
        }
    elif claim["mainsnak"]["datatype"] == "external-id":
        return {
            "type": ClaimType.STRING,
            "value": claim["mainsnak"]["datavalue"]["value"],
        }
    elif claim["mainsnak"]["datatype"] == "monolingualtext":
        return {
            "type": ClaimType.STRING,
            "value": claim["mainsnak"]["datavalue"]["value"]["text"],
        }
    elif claim["mainsnak"]["datatype"] == "url":
        return {
            "type": ClaimType.STRING,
            "value": claim["mainsnak"]["datavalue"]["value"],
        }
    elif claim["mainsnak"]["datatype"] == "quantity":
        # TODO: handle parseing of value and unit
        return {
            "type": ClaimType.QUANTITY,
            "value": claim["mainsnak"]["datavalue"]["value"]["amount"],
        }
    elif claim["mainsnak"]["datatype"] == "wikibase-lexeme":
        return {
            "type": ClaimType.LEXEME,
            "value": claim["mainsnak"]["datavalue"]["value"]["id"],
        }
    elif claim["mainsnak"]["datatype"] == "wikibase-form":
        return {
            "type": ClaimType.LEXEME,
            "value": claim["mainsnak"]["datavalue"]["value"]["id"],
        }
    elif claim["mainsnak"]["datatype"] == "geo-shape":
        return {"type": ClaimType.STRING, "value": claim["mainsnak"]["datavalue"]["value"]}
    elif claim["mainsnak"]["datatype"] == "wikibase-property":
        return {"type": ClaimType.PROPERTY, "value": claim["mainsnak"]["datavalue"]["value"]["id"]}
    elif claim["mainsnak"]["datatype"] == "commonsMedia" or claim["mainsnak"]["datatype"] == "globe-coordinate":
        # ignore
        return None
    else:
        print(f"Unknown claim type: {claim['mainsnak']['datatype']}")
        print(f"Claim: {claim}")
        return {
            "type": ClaimType.UNKNOWN,
            "value": claim["mainsnak"]["datavalue"]["value"],
        }


def get_entity_claims(id):
    params = {
        "action": "wbgetentities",
        "languages": "en",
        "format": "json",
        "ids": f"Q{id}",
        "props": "claims",
    }

    data = requests.get(API_URL, params=params)
    res = data.json()["entities"][f"Q{id}"]["claims"]

    claims = {}
    for property in res:
        for claim in res[property]:
            parsed_claim = parse_claim(claim)
            if parsed_claim:
                claims[property] = parsed_claim

    return claims


mention = input("Enter a mention: ")
candidates = get_candidates(mention)

from pprint import pprint

for candidate in candidates:
    print(f"Q{candidate}:")
    pprint(get_entity_claims(candidate))
    print("")
