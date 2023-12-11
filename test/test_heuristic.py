import numpy as np

from entsearch.search import SearchTree
from entsearch.data import sample_dirichlet

rng = np.random.default_rng(seed=1000)

m = 10
alpha = np.ones(m)
proba = sample_dirichlet(alpha, generator=rng)

n = 1_000
y_cat = rng.choice(m, size=n, p=proba)
y_one_hot = np.zeros((n, m))
y_one_hot[np.arange(n), y_cat] = 1

model = SearchTree(m, comeback=True)

for i, y in enumerate(y_cat):
    assert model.root.value == i
    model.coarse_identification(y, epsilon=0)

    # counts = y_one_hot[:i+1].sum(axis=0)
    huffman_list = model.get_huffman_list()
    assert huffman_list[model._i_part:] == model.huffman_list[model._i_part:]
    old = huffman_list[model._i_part]
    for node in huffman_list[model._i_part + 1:]:
        assert not node < old
        old = node

out_str = "                                           Node: 1000                                                                       \n                 Node:  438                     |        Node:  562                                                         \n     Node:  217       |       Node:  221        | \x1b[1mLeaf 0: 229\x1b[0m |        Node:  333                                           \n\x1b[1mLeaf 5: 0\x1b[0m | \x1b[1mLeaf 2: 0\x1b[0m | \x1b[1mLeaf 4: 57\x1b[0m | \x1b[1mLeaf 3: 72\x1b[0m |             | \x1b[1mLeaf 1: 146\x1b[0m |       Node:  187                              \n                                                |             |             | \x1b[1mLeaf 7: 11\x1b[0m |      Node:   15                  \n                                                |             |             |            | \x1b[1mLeaf 8: 2\x1b[0m |      Node:   10      \n                                                |             |             |            |           | \x1b[1mLeaf 6: 1\x1b[0m | \x1b[1mLeaf 9: 1\x1b[0m\n"

assert model.__str__() == out_str
