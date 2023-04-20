from util import pickle_load


x = pickle_load("best-model-so-far-8055", is_dump=True)

print(x["f1"])
print(x["params"])