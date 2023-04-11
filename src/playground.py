from util import pickle_load, pickle_save

# x1 = pickle_load("all-test-tag", is_dump=True)

# # get tags and tag ratio
# for col in x1:
#     print(f"Column: {col.cells}")
#     for cell in col.cells:
#         print(f"Cell: {cell}")
#         print()
#         for candidate in cell.candidates:
#             print(f"Candidate: {candidate.to_sentence}")
#             print(f"Tag:    {candidate.tag}")
#             print(f"Tag ratio:    {candidate.tag_ratio}")
#             print()

x2 = pickle_load("best-params", is_dump=True)

print(x2.get_params())
