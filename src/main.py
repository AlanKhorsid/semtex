from typing import Union
import requests
import threading
from enum import Enum
from _types import ClaimType, Entity, Claim, DiscoveredEntity
from pprint import pprint

from classes import Candidate, CandidateSet
from _requests import wikidata_get_entity
from util import parse_entity_properties, open_dataset

API_URL = "https://www.wikidata.org/w/api.php"

HASH: dict[set[str]] = {}


def get_candidates(mention: str, limit: int = 10) -> list[Entity]:
    """
    Fetches a list of candidate entities from the Wikidata API.

    Parameters
    ----------
    mention : str
        The mention to search for.
    limit : int, optional
        The maximum number of candidates to return, by default 10

    Returns
    -------
    list[Entity]
        A list of candidate entities.

    Example
    -------
    >>> get_candidates("Barack Obama", limit=3)
    [{'id': 'Q76', 'claims': {}}, {'id': 'Q47513588', 'claims': {}}, {'id': 'Q59661289', 'claims': {}}]
    """

    params = {
        "action": "wbsearchentities",
        "language": "en",
        "format": "json",
        "search": mention,
        "limit": f"{limit}",
    }
    data = requests.get(API_URL, params=params)

    res: list[Entity] = []
    for entity in data.json()["search"]:
        id: str = entity["id"]
        res.append({"id": id, "claims": {}})

    return res


def parse_claim(claim: dict) -> Union[Claim, None]:
    """
    Parses a claim from a JSON resonse from the Wikidata API.

    For now, only claims of type "wikibase-item" are parsed. These claims map to actual entities.


    """

    if claim["mainsnak"]["snaktype"] == "novalue" or claim["mainsnak"]["snaktype"] == "somevalue":
        return None

    if claim["mainsnak"]["datatype"] == "wikibase-item":
        if claim["mainsnak"]["property"] != "P31" and claim["mainsnak"]["datavalue"]["value"]["id"] != "P279":
            return None

        return {
            "type": ClaimType.ENTITY,
            "value": claim["mainsnak"]["datavalue"]["value"]["id"],
        }
    else:
        return None

    if claim["mainsnak"]["datatype"] == "wikibase-item":
        return {
            "type": ClaimType.ENTITY,
            "value": claim["mainsnak"]["datavalue"]["value"]["id"],
        }
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
        return {
            "type": ClaimType.STRING,
            "value": claim["mainsnak"]["datavalue"]["value"],
        }
    elif claim["mainsnak"]["datatype"] == "wikibase-property":
        return {
            "type": ClaimType.PROPERTY,
            "value": claim["mainsnak"]["datavalue"]["value"]["id"],
        }
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


def expand_entity(
    entity: Entity,
    trace: str = "",
    src_entity: Union[Entity, None] = None,
    num_threads=50,
) -> Entity:
    if entity["claims"] != {}:
        threads = []
        for e in entity["claims"]:
            t = threading.Thread(
                target=expand_entity,
                args=(
                    entity["claims"][e],
                    f"{trace}{entity['id']} -> ",
                    src_entity or entity,
                ),
            )
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
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


def candidates_iter(candidate_sets: list[CandidateSet], skip_index: int = -1) -> list[Candidate]:
    for i, candidate_set in enumerate(candidate_sets):
        if i == skip_index:
            continue
        for candidate in candidate_set.candidates:
            yield candidate


dataset = open_dataset(correct_spelling=True)

# Fetch candidates
all_candidates: list[CandidateSet] = []
for (mention, id) in dataset[:5]:
    candidate_set = CandidateSet(mention, correct_id=id)
    candidate_set.fetch_candidates()
    candidate_set.fetch_candidate_info()
    all_candidates.append(candidate_set)

# Generate features and labels
data = []
labels = []
for i, candidate_set in enumerate(all_candidates):
    for candidate in candidate_set.candidates:
        instance_total = 0
        instance_overlap = 0
        subclass_total = 0
        subclass_overlap = 0
        description_overlaps = []

        for other_candidate in candidates_iter(all_candidates, i):
            (overlap, total) = candidate.instance_overlap(other_candidate)
            instance_total += total
            instance_overlap += overlap

            (overlap, total) = candidate.subclass_overlap(other_candidate)
            subclass_total += total
            subclass_overlap += overlap

            description_overlaps.append(candidate.description_overlap(other_candidate))

        labels.append(candidate.is_correct)
        data.append(
            [
                # candidate.title,
                # candidate.description,
                candidate.lex_score(candidate_set.mention),
                instance_overlap / instance_total if instance_total > 0 else 0,
                subclass_overlap / subclass_total if subclass_total > 0 else 0,
                sum(description_overlaps) / len(description_overlaps) if len(description_overlaps) > 0 else 0,
            ]
        )

pprint(data)
pprint(labels)
