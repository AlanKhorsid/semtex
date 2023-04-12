from util import (
        progress,
        pickle_save,
        pickle_load,
        )
from classes import (Column)
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

    for cand in candidates:
        for instance in cand.instances:
            if instance in typedict:
                typedict[instance] = typedict[instance] + 1
            else:
                typedict[instance] = 1

    for key, value in typedict.items():
        if value == len(candidates):
            return key

    return cta_query(typedict)


def cta_query(typedict):
    pass

#PICKLE_FILE_NAME = "test-2022-bing"
#cols: list[Column] = pickle_load(f"{PICKLE_FILE_NAME}", is_dump=True)
