from util import pickle_load, pickle_save


x = pickle_load("best-model-so-far", is_dump=True)

print(x["f1"])
print(x["params"])
x = 0
