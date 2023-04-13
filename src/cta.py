from util2 import (
        pickle_load,
        )
from classes2 import Candidate
from _requests import wikidata_entity_search

#----CTA PLAN----#
# input: list of candidates from CEA
# check if the candidates have the same instance of
#   if true, instance of is CTA
#   else, make a list of instance of, check each instance of's subclass to see if it's in the list, otherwise add to list and keep going
#
# maybe set limit to how deep we check?
#   if limit is hit, the largest overlap becomes the type (in case of errors in CEA)
#
# output: common type annotation

def cta_no_query(candidates):
    typedict = dict()
    best_value = 0
    best_key = 0
    amount_of_nonetypes = 0

    for candidate in candidates:
        if candidate == None:
            amount_of_nonetypes += 1
            continue
        for statement in candidate.statements:
            if statement.property == 31:
                #print(f"statement value: {statement.value}")
                if statement.value in typedict:
                    typedict[statement.value] = typedict[statement.value] + 1
                else:
                    typedict[statement.value] = 1

    for key, value in typedict.items():
        if value > best_value:
            best_key = key
            best_value = value
    print(f"return key: {best_key} confidence: {best_value/len(candidates)} percentage Nonetypes: {amount_of_nonetypes/len(candidates)}")
    if amount_of_nonetypes == len(candidates):
        print("=====ALL NONTYPES=====")
        return None

    return best_key

if __name__ == "__main__":
    PICKLE_FILE_NAME = "cea_results"
    candidates: list[Candidate] = pickle_load(f"{PICKLE_FILE_NAME}", is_dump=False)
    successes = 0
    failures = 0
    for cand in candidates:
        result = cta_no_query(cand['candidates'])
        if result == cand['cta']:
            print(f"correct result found: {cand['cta']}")
            successes += 1
        else:
            print("======DID NOT FIND INSTANCE======")
            print(f"expected: {cand['cta']} result: {result}")
            failures += 1
    print(f"successes: {successes}   failures: {failures}")
    print(f"success percentage: {successes/(successes+failures)}")
