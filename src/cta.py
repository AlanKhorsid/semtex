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


#TODO: rename cta_no_query to cta
def cta_no_query(candidates, correct = None, print_output = False, max_depth = 1, explore_tree = False):
    instance_dict = dict()
    best_value = 0
    best_key = 0
    correct_key = None
    amount_of_nonetypes = 0
    current_depth = 0
    confidence = 0

    #get instance_dict and amount of 
    instance_dict, amount_of_nonetypes = add_to_instance_dict(candidates, instance_dict)

    best_key, best_value, correct_key = find_best_key(instance_dict, correct=correct) 

    confidence = best_value/(len(candidates)-amount_of_nonetypes)
    if confidence > 1:
        print("!!!!!WHAT!!!!!")

    while confidence != 1 and current_depth < max_depth and explore_tree == True:
        current_depth += 1
        #query wikidata on subclass of and part of on each key in instance_dict
        instance_dict = cta_query(instance_dict)
        best_key, best_value, correct_key = find_best_key(instance_dict, correct=correct)
        confidence = best_value/(len(candidates)-amount_of_nonetypes)
    

    if print_output:
        print(f"return key: {best_key} confidence: {confidence} best-value/len(candidates): {best_value/len(candidates)} percentage Nonetypes: {amount_of_nonetypes/len(candidates)}")
        if amount_of_nonetypes == len(candidates):
            print("=====ALL NONTYPES=====")
            return None
        if correct_key != best_key and correct_key != None:
            print(f"Found the correct key in the dataset, it is not the return key: {correct_key}")
            print(f"return key: {best_key} confidence: {confidence} best-value/len(candidates): {best_value/len(candidates)} percentage Nonetypes: {amount_of_nonetypes/len(candidates)}")

    return best_key

def cta_query(instance_dict : dict):
    fetch_ids = []
    candidates = []
    for key, value in instance_dict.items():
        fetch_ids.append(key)

    wikidata_fetch_entities(fetch_ids)
    for ids in fetch_ids:
        candidates.append(Candidate(ids))

    for cand in candidates:
        cand.fetch_info()

    add_to_instance_dict(candidates, instance_dict, tree_exploration = True)

    return instance_dict

def add_to_instance_dict(candidates, instance_dict, tree_exploration = False):
    amount_of_nonetypes = 0
    for candidate in candidates:
        if candidate == None:
            amount_of_nonetypes += 1
            continue
        for statement in candidate.statements:
            if ((statement.property == 31 and tree_exploration == False) or (statement.property == 279 and tree_exploration == True)):
                #print(f"statement value: {statement.value}")
                if statement.value in instance_dict:
                    instance_dict[statement.value] = instance_dict[statement.value] + 1
                else:
                    instance_dict[statement.value] = 1
    return instance_dict, amount_of_nonetypes

def find_best_key(instance_dict, correct = None):
    best_key = None
    best_value = 0
    correct_key = None
    for key, value in instance_dict.items():
        if value > best_value:
            best_key = key
            best_value = value
        if correct != None and key == correct:
            correct_key = key

    return best_key, best_value, correct_key


if __name__ == "__main__":
    #cta_query({618123: 1})

    PICKLE_FILE_NAME = "cea_results_test"
    candidates: list[Candidate] = pickle_load(f"{PICKLE_FILE_NAME}", is_dump=False)
    candidates = candidates[0:200]
    successes = 0
    failures = 0
    for cand in candidates:
        result = cta_no_query(cand['candidates'], correct=cand['cta'], print_output=True, explore_tree = True)
        if result == cand['cta']:
            successes += 1
        else:
            print("======DID NOT FIND INSTANCE======")
            print(f"expected: {cand['cta']} result: {result}")
            failures += 1
    print(f"successes: {successes}   failures: {failures}")
    print(f"success percentage: {successes/(successes+failures)}")
