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
