from util import pickle_load, pickle_save

x1 = pickle_load("with-semantic-features-test-0-2000", is_dump=True)
x2 = pickle_load("with-semantic-features-test-2000-4000", is_dump=True)
x3 = pickle_load("with-semantic-features-test-4000-5600", is_dump=True)
x4 = pickle_load("with-semantic-features-test-5601-5631", is_dump=True)
x5 = pickle_load("with-semantic-features-test-5631", is_dump=True)

# for x in x5.cells:
#     for cand in x.candidates:
#         if hasattr(cand, "most_similar_to"):
#             print(cand.most_similar_to)
#         else:
#             print("x5 None")

# z = pickle_load("all-test-tag", is_dump=True)

# remove the first 2000 from x2
x2 = x2[2000:]
# remove the first 4000 from x3
x3 = x3[4000:]


# print(len(x1) + len(x2) + len(x3) + len(x4))

x = x1 + x2 + x3 + x4
x.append(x5)
pickle_save(x, "all-test-with-semantic-features")
# list_x = []
# list_x_cells = []
# list_z = []
# list_z_cells = []
# # check if x and z are the same and in the same order. If not then print the column and the cells
# for col in x:
#     list_x_cells = []
#     for cell in col.cells:
#         list_x_cells.append(cell.mention)

#     list_x.append(list_x_cells)


# for col in z:
#     list_z_cells = []
#     for cell in col.cells:
#         list_z_cells.append(cell.mention)

#     list_z.append(list_z_cells)

# num = 0
# # check if x and z are the same and in the same order. If not then print the column and the cells
# for i in range(len(list_x)):
#     if list_x[i] != list_z[i]:
#         print(f"Column {i} is not the same")
#         print(f"X: {list_x[i]}")
#         print(f"Z: {list_z[i]}")
#         num += 1

# print(num)
