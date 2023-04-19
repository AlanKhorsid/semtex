from util2 import (
        pickle_load,
        )
from classes2 import Candidate
from _requests import wikidata_fetch_entities, get_entity

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

def cta_no_query(candidates, correct=None, print_output=False):
    typedict = dict()
    best_value = 0
    best_key = 0
    correct_key = None
    amount_of_nonetypes = 0
    confidence = 0

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
        if correct != None and key == correct:
            correct_key = key

    confidence = best_value/(len(candidates)-amount_of_nonetypes)

    while confidence != 1:
        fetch_ids = []
        #query wikidata on subclass of and part of on each key in typedict
        for key, value in typedict.items():
            #add key to fetch list
            fetch_ids.append(key)

        break

    if print_output:
        print(f"return key: {best_key} confidence: {confidence} best-value/len(candidates): {best_value/len(candidates)} percentage Nonetypes: {amount_of_nonetypes/len(candidates)}")
        if amount_of_nonetypes == len(candidates):
            print("=====ALL NONTYPES=====")
            return None
        if correct_key != best_key and correct_key != None:
            print(f"Found the correct key in the dataset, it is not the return key: {correct_key}")
            print(f"return key: {best_key} confidence: {confidence} best-value/len(candidates): {best_value/len(candidates)} percentage Nonetypes: {amount_of_nonetypes/len(candidates)}")

    return best_key

if __name__ == "__main__":
    PICKLE_FILE_NAME = "cea_results_test"
    candidates: list[Candidate] = pickle_load(f"{PICKLE_FILE_NAME}", is_dump=False)
    successes = 0
    failures = 0
    for cand in candidates:
        result = cta_no_query(cand['candidates'], correct=cand['cta'], print_output=True)
        if result == cand['cta']:
            successes += 1
        else:
            print("======DID NOT FIND INSTANCE======")
            print(f"expected: {cand['cta']} result: {result}")
            failures += 1
    print(f"successes: {successes}   failures: {failures}")
    print(f"success percentage: {successes/(successes+failures)}")
