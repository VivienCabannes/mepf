
import numpy as np

from entsearch.deterministic import SearchTree
from entsearch.binary_tree import Node
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


rng = np.random.default_rng(seed=0)

# generate data
m = 10
n = 10000
alpha = np.zeros(m) + 1
proba = sample_dirichlet(alpha, generator=rng)
y_cat = rng.choice(m, size=n, p=proba)

model = SearchTree(m)

model.adaptive = False
for y in y_cat:
    model.exhaustive_search(y)
nb_queries_dichotomic = model.nb_queries

reset_value(model)
model.nb_queries = 0
model.adaptive = True
for i, y in enumerate(y_cat):
    model.exhaustive_search(y)
nb_queries_adaptive = model.nb_queries

reset_value(model)
model.nb_queries = 0
model.adaptive = False
for y in y_cat:
    model.exhaustive_search(y)
nb_queries_huffman = model.nb_queries

model = SearchTree(m)
model.batch_search(y_cat)
nb_queries_batch = model.nb_queries

for _ in range(10):
    model.nb_queries = 0
    model.batch_search(y_cat)
nb_queries_best_batch = model.nb_queries

model = SearchTree(m, comeback=True, adaptive=True)
from entsearch import TruncatedSearch
model = TruncatedSearch(m, comeback=True)
for y in y_cat:
    # model.truncated_search(y, epsilon=0)
    model(y, epsilon=0)
nb_queries_coarse = model.nb_queries

for _ in range(10):
    model.nb_queries = 0
    for y in y_cat:
        model(y, epsilon=0)
        # model.truncated_search(y, epsilon=0)
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
print("Dicho expected", dicho_pred, dicho_pred + n)
print("Huffman expected", huffman_pred, huffman_pred + n)
print("Coarse expected", coarse_pred, coarse_pred + n)
