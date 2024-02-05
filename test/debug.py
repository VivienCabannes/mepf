
import numpy as np

from entsearch.search import SearchTree
from entsearch.search.tree_constructors import Node
from entsearch.data import sample_dirichlet


def reset_value(model):
    to_reset = [model.root]
    while len(to_reset):
        node = to_reset.pop()
        node.value = 0
        if type(node) is Node:
            to_reset.append(node.right)
            to_reset.append(node.left)
    model.partition = list(model.y2leaf.values())
    model._i_part = 0
    model.y2node = model.y2leaf.copy()


rng = np.random.default_rng(seed=1000)

# generate data
m = 10
n = 100000
alpha = np.zeros(m) + 100
proba = sample_dirichlet(alpha, generator=rng)
y_cat = rng.choice(m, size=n, p=proba)

model = SearchTree(m)

for y in y_cat:
    model.fine_identification(y, update=False)
nb_queries_dichotomic = model.nb_queries

reset_value(model)
model.nb_queries = 0
for i, y in enumerate(y_cat):
    model.fine_identification(y, update=True)
nb_queries_adaptive = model.nb_queries

reset_value(model)
model.nb_queries = 0
for y in y_cat:
    model.fine_identification(y, update=False)
nb_queries_huffman = model.nb_queries

model = SearchTree(m)
model.batch_identification(y_cat, update=True)
nb_queries_batch = model.nb_queries

for _ in range(10):
    model.nb_queries = 0
    model.batch_identification(y_cat, update=True)
nb_queries_best_batch = model.nb_queries

model = SearchTree(m, comeback=True)
for y in y_cat:
    model.coarse_identification(y, epsilon=0)
nb_queries_coarse = model.nb_queries

for _ in range(10):
    model.nb_queries = 0
    for y in y_cat:
        model.coarse_identification(y, epsilon=0)
nb_queries_best_coarse = model.nb_queries

print(f"nb_queries_dichotomic: {nb_queries_dichotomic}")
print(f"nb_queries_adaptive: {nb_queries_adaptive}")
print(f"nb_queries_huffman: {nb_queries_huffman}")
print(f"nb_queries_batch: {nb_queries_batch}")
print(f"nb_queries_best_batch: {nb_queries_best_batch}")
print(f"nb_queries_coarse: {nb_queries_coarse}")
print(f"nb_queries_best_coarse: {nb_queries_best_coarse}")

dicho_pred = int(np.log2(m) * n)
huffman_pred = int(-(np.log2(proba) * proba).sum() * n)
coarse_pred = int(-np.log2(proba[0]) * n)
print("Dicho expected", dicho_pred)
print("Huffman expected", huffman_pred)
print("Coarse expected", coarse_pred)
