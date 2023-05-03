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


#TODO: Make cta_retriever check if a result is an instance of of another equally good result
def cta_retriever(candidates_list, correct = None, max_depth = 0):
    result_ids = []
    best_values = []
    instance_dicts = []
    confidence_vector = []
    correct_key_vector = []
    ids_to_query = []
    current_depth = 0
    index = 0
    for i in range(len(candidates_list)):
        #initialize lists with base values
        result_ids.append(None)
        best_values.append(0)
        instance_dicts.append(dict())
        confidence_vector.append(0)
        if correct == None:
            correct_key_vector.append(None)
        else:
            correct_key_vector = correct

    #while loop to collect and query for instace of elements
    while index < len(candidates_list) and current_depth <= max_depth:
        print(ids_to_query)
        wikidata_fetch_entities(ids_to_query)
        ids_to_query = []
            
        for candidates in candidates_list:
            best_key = result_ids[index]
            correct_key = correct_key_vector[index]
            amount_of_nonetypes = 0
            confidence = 0
            explore_next = []

            #get instance_dict and amount of nones in first pass
            if current_depth == 0:
                instance_dicts[index], amount_of_nonetypes, explore_next = add_to_instance_dict(candidates, instance_dicts[index])

                best_key, best_values[index], correct_key = find_best_key(instance_dicts[index], correct_key=correct_key) 

                confidence_vector[index] = best_values[index]/(len(candidates)-amount_of_nonetypes)
            
                #exception in case we have a confidence higher than 100%
                if confidence_vector[index] > 1:
                    raise ValueError()
            else:
                cands = fetch_cand_info(instance_dicts[index])
                instance_dicts[index], num_of_none, explore_next = add_to_instance_dict(cands, instance_dicts[index], tree_exploration = True)

                #placeholder vars for previously best key and value to compare with the next iteration's values
                current_best_value = best_values[index]
                current_best_key = best_key

                best_key, best_values[index], correct_key = find_best_key(instance_dicts[index], correct_key=correct)

                confidence_vector[index] = best_values[index]/(len(candidates)-amount_of_nonetypes)

            
                if confidence_vector[index] <= current_best_value/(len(candidates)-amount_of_nonetypes):
                    best_values[index] = current_best_value
                    best_key = current_best_key
                    confidence_vector[index] = current_best_value/(len(candidates)-amount_of_nonetypes)
            
            #Add ids of the entites that should be explored next iteration
            for val in explore_next:
                if val not in ids_to_query:
                    ids_to_query.append(val)
            
            #set the result id as the best found key
            result_ids[index] = best_key
            index += 1

            #debug output
            if correct != None:
                if amount_of_nonetypes == len(candidates):
                    print("=====ALL NONTYPES=====")
                    best_key = 0
                if correct_key != best_key and correct_key != None:
                    print(f"Found the correct key in the dataset, it is not the return key: {correct_key}")
                    print(f"return key: {best_key} confidence: {confidence} best-value/len(candidates): {best_values[index]/len(candidates)} percentage Nonetypes: {amount_of_nonetypes/len(candidates)}")

        #prepare for next iteration of while
        current_depth += 1
        index = 0

    return result_ids


def fetch_cand_info(instance_dict : dict):
    candidates = []
    num_of_none = 0

    for key, value in instance_dict.items():
        candidates.append(Candidate(key))
    
    for cand in candidates:
        cand.fetch_info()

    return candidates

def add_to_instance_dict(candidates, instance_dict, tree_exploration = False):
    amount_of_nonetypes = 0
    explore_next = []
    for candidate in candidates:
        discovered_ids = []
        if candidate == None:
            amount_of_nonetypes += 1
            continue

        for statement in candidate.statements:
            if ((statement.property == 31 and tree_exploration == False) or (statement.property == 279 and tree_exploration == True)):
                if statement.value in discovered_ids:
                    continue


                if statement.value in instance_dict:
                    instance_dict[statement.value] = instance_dict[statement.value] + 1

                else:
                    instance_dict[statement.value] = 1
                
                discovered_ids.append(statement.value)
        
        for ids in discovered_ids:
            if ids not in explore_next:
                explore_next.append(ids)
        discovered_ids = []

    return instance_dict, amount_of_nonetypes, explore_next

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


#test code
if __name__ == "__main__":
    PICKLE_FILE_NAME = "cea_results_test"
    candidates_objects: list[Candidate] = pickle_load(f"{PICKLE_FILE_NAME}", is_dump=False)
    candidates_objects = candidates_objects
    candidates = []
    correct_results = []
    successes = 0
    failures = 0
    max_depth_failures = 0
    max_depth_successes = 0
    index = 0

    for cand in candidates_objects:
        candidates.append(cand['candidates'])
        correct_results.append(cand['cta'])

    result1 = cta_retriever(candidates)
    result2 = cta_retriever(candidates, max_depth = 1, correct=correct_results)

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
            print("entity values:")
            for c in candidates[index]:
                print(c.id)
        elif result1[index] != result2[index] and result2[index] == cand['cta']:
            max_depth_successes += 1
        index += 1
    print(f"successes: {successes}   failures: {failures}")
    print(f"max depth 1 successes: {max_depth_successes}   failures: {max_depth_failures}")
    print(f"max depth 0 success percentage: {successes/(successes+failures)}")
    print(f"max depth 1 success percentage: {max_depth_successes/(max_depth_successes+max_depth_failures)}")
    print(f"max depth successes: {max_depth_successes}   max depth failures: {max_depth_failures}   success/failures: {max_depth_successes/max_depth_failures}")
