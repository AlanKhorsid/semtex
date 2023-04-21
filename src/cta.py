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
def cta_retriever(candidates_list, correct = None, print_output = False, max_depth = 0):
    result_ids = []
    instance_dicts = []
    confidence_vector = []
    correct_key_vector = []
    index = 0
    for i in range(len(candidates_list)):
        result_ids.append(None)
        instance_dicts.append(dict())
        confidence_vector.append(0)
        if correct == None:
            correct_key_vector.append(None)
        else:
            correct_key_vector = correct



    for candidates in candidates_list:
        best_value = 0
        best_key = 0
        correct_key = correct_key_vector[index]
        amount_of_nonetypes = 0
        current_depth = 0
        confidence = 0

        #get instance_dict and amount of 
        instance_dicts[index], amount_of_nonetypes = add_to_instance_dict(candidates, instance_dicts[index])

        best_key, best_value, correct_key = find_best_key(instance_dicts[index], correct_key=correct_key) 

        confidence = best_value/(len(candidates)-amount_of_nonetypes)
        if confidence > 1:
            print(f"index: {index}, best_value: {best_value}, dict: {instance_dicts[index]}, candidates length: {len(candidates)}")
            for cand in candidates:
                print(cand.id)
            raise ValueError()

        while confidence < 1 and current_depth < max_depth:
            current_depth += 1
            current_best_value = best_value
            current_best_key = best_key
            #query wikidata on subclass of and part of on each key in instance_dict
            instance_dicts[index] = cta_query(instance_dicts[index])
            best_key, best_value, correct_key = find_best_key(instance_dicts[index], correct=correct)
            confidence = best_value/(len(candidates)-amount_of_nonetypes)
            if confidence < current_best_value/(len(candidates)-amount_of_nonetypes):
                best_value = current_best_value
                best_key = current_best_key
        

        if print_output:
            if amount_of_nonetypes == len(candidates):
                print("=====ALL NONTYPES=====")
                best_key = 0
            if correct_key != best_key and correct_key != None:
                print(f"Found the correct key in the dataset, it is not the return key: {correct_key}")
                print(f"return key: {best_key} confidence: {confidence} best-value/len(candidates): {best_value/len(candidates)} percentage Nonetypes: {amount_of_nonetypes/len(candidates)}")
        result_ids[index] = best_key
        if result_ids[index] == 0:
            print("HAD ZERO VALUE")

        index += 1

    return result_ids


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
        discovered_ids = []
        if candidate == None:
            amount_of_nonetypes += 1
            continue

        for statement in candidate.statements:
            if ((statement.property == 31 and tree_exploration == False) or (statement.property == 279 and tree_exploration == True)):
                #print(f"statement value: {statement.value}")
                if statement.value in discovered_ids:
                    #print(f"continue called statement_value: {statement.value}")
                    continue


                if statement.value in instance_dict:
                    instance_dict[statement.value] = instance_dict[statement.value] + 1

                else:
                    instance_dict[statement.value] = 1
                
                discovered_ids.append(statement.value)

        discovered_ids = []

    return instance_dict, amount_of_nonetypes

def find_best_key(instance_dict, correct_key = None):
    best_key = None
    best_value = 0
    result_key = None
    for key, value in instance_dict.items():
        if value > best_value:
            best_key = key
            best_value = value
        if correct_key != None and key == correct_key:
            result_key = key

    return best_key, best_value, result_key


if __name__ == "__main__":
    #cta_query({618123: 1})

    PICKLE_FILE_NAME = "cea_results_test"
    candidates_objects: list[Candidate] = pickle_load(f"{PICKLE_FILE_NAME}", is_dump=False)
    candidates_objects = candidates_objects[0:100]
    candidates = []
    successes = 0
    failures = 0
    max_depth_failures = 0
    max_depth_successes = 0
    index = 0

    for cand in candidates_objects:
        candidates.append(cand['candidates'])

    result1 = cta_retriever(candidates)
    result2 = cta_retriever(candidates)

    print(result1)

    for cand in candidates_objects:
        if result1[index] == cand['cta']:
            successes += 1
        else:
            print("======DID NOT FIND INSTANCE======")
            print(f"expected: {cand['cta']} result: {result1[index]}")
            failures += 1
        if result1[index] != result2[index] and result1[index] == cand['cta']:
            max_depth_failures += 1
            print(f"max depth = 1 failed when max depth = 0 didn't")
            print(f"expected: {cand['cta']} result: {result2[index]}")
        elif result1[index] != result2[index] and result2[index] == cand['cta']:
            max_depth_successes += 1
        index += 1
    print(f"successes: {successes}   failures: {failures}")
    print(f"success percentage: {successes/(successes+failures)}")
    print(f"max depth successes: {max_depth_successes}   max depth failures: {max_depth_failures}")
