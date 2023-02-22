from typing import Union
import requests
from enum import Enum
from _types import ClaimType, Entity, Claim, DiscoveredEntity
from pprint import pprint

API_URL = "https://www.wikidata.org/w/api.php"

HASH: dict[set[str]] = {}


def get_candidates(mention: str) -> list[Entity]:
    params = {
        "action": "wbsearchentities",
        "language": "en",
        "format": "json",
        "search": mention,
        "limit": "10",
    }
    data = requests.get(API_URL, params=params)

    res: list[Entity] = []
    for entity in data.json()["search"]:
        id: str = entity["id"]
        res.append({"id": id, "claims": {}})

    return res


def parse_claim(claim) -> Union[Claim, None]:
    if claim["mainsnak"]["snaktype"] == "novalue" or claim["mainsnak"]["snaktype"] == "somevalue":
        return None

    if claim["mainsnak"]["datatype"] == "wikibase-item":
        if claim["mainsnak"]["property"] != "P31" and claim["mainsnak"]["datavalue"]["value"]["id"] != "P279":
            return None

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


def get_entity_claims(entity: Entity) -> list[Claim]:
    params = {
        "action": "wbgetentities",
        "languages": "en",
        "format": "json",
        "ids": entity["id"],
        "props": "claims",
    }

    data = requests.get(API_URL, params=params)
    res = data.json()["entities"][entity["id"]]["claims"]

    claims = []
    for property in res:
        property: str = property
        for claim in res[property]:
            parsed_claim = parse_claim(claim)
            if parsed_claim:
                claims.append(parsed_claim)

    return claims


def expand_entity(entity: Entity, trace: str = "", src_entity: Union[Entity, None] = None) -> Entity:
    if entity["claims"] != {}:
        for e in entity["claims"]:
            expand_entity(entity["claims"][e], f"{trace}{entity['id']} -> ", src_entity or entity)
        return

    print(f"Expanding {trace}{entity['id']}")
    claims = get_entity_claims(entity)

    for claim in claims:
        entity["claims"][claim["value"]] = {"id": claim["value"], "claims": {}}

        if claim["value"] not in HASH:
            HASH[claim["value"]] = set()
            HASH[claim["value"]].add(src_entity["id"] if src_entity else entity["id"])
        else:
            HASH[claim["value"]].add(src_entity["id"] if src_entity else entity["id"])


def candidate_index(candidatesList: list[list[Entity]], candidate: str) -> int:
    for i in range(len(candidatesList)):
        for entity in candidatesList[i]:
            if entity["id"] == candidate:
                return i
    return -1


def get_candidate_coverage(
    hash: dict[set[str]], candidatesList: list[list[Entity]]
) -> list[(str, float, list[(int, list[str])])]:
    num_indices = len(candidatesList)

    res = []
    for entity in hash:
        num_indices_covered = 0

        cands: list[(int, list[str])] = []

        for candidate in hash[entity]:
            candidate: str = candidate

            i = candidate_index(candidatesList, candidate)
            if i == -1:
                raise Exception(f"Candidate {candidate} not found in candidatesList. This should not happen?")

            if not any(i == cand[0] for cand in cands):
                cands.append((i, [candidate]))
                num_indices_covered += 1
            else:
                for cand in cands:
                    if cand[0] == i:
                        cand[1].append(candidate)
                        break

        res.append((entity, num_indices_covered / num_indices, cands))

    res.sort(key=lambda x: x[1], reverse=True)

    # remove all results that are not 100% covered
    # res = [x for x in res if x[1] == 1]

    return res


# MENTIONS = ["Helgafell", "Tungurahua volcano", "Khodutka", "Gamchen", "Voyampolsky"]
MENTIONS = ["Barack Obama", "Donald Trump", "Joe Biden", "Hillary Clinton", "Bernie Sanders"]
candidatesList = [get_candidates(mention) for mention in MENTIONS]

for candidates in candidatesList:
    for candidate in candidates:
        expand_entity(candidate)

for candidates in candidatesList:
    for candidate in candidates:
        expand_entity(candidate)

for candidates in candidatesList:
    for candidate in candidates:
        expand_entity(candidate)

# expand_entity(candidatesList[0][0])
# expand_entity(candidatesList[1][0])
# expand_entity(candidatesList[2][0])
# expand_entity(candidatesList[3][0])
# expand_entity(candidatesList[4][0])
# expand_entity(candidatesList[0][0])

# pprint(HASH)
cov = get_candidate_coverage(HASH, candidatesList)
pprint(cov)
pprint(len([x for x in cov if x[1] == 1]))

# EXPAND ALL CANDIDATES
# for candidates in candidatesList:
#     for candidate in candidates:
#         expand_entity(candidate)
