from util import pickle_load


x1 = pickle_load("test-2022-bing-tag-1", is_dump=True)

# print every tag
for col in x1:
    for cell in col.cells:
        for c in cell.candidates:
            print(c.tag)
