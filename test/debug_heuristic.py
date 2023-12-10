
import numpy as np

from entsearch.search import HeuristicSearch
from entsearch.data import sample_dirichlet

rng = np.random.default_rng(seed=1000)

m = 10
alpha = np.ones(m)
proba = sample_dirichlet(alpha, generator=rng)

n = 1_000
y_cat = rng.choice(m, size=n, p=proba)
y_one_hot = np.zeros((n, m))
y_one_hot[np.arange(n), y_cat] = 1

model = HeuristicSearch([0 for _ in range(m)])

for i, y in enumerate(y_cat):
    model.report_observation(y)
    counts = y_one_hot[:i+1].sum(axis=0)
    for z in range(m):
        if model.y2leaf[z].value > counts[z] + 1:
            print('Error')

out_str = "                               Node: 1000                                                                                    \n                 Node:  434         |                     Node:  566                                                         \n     Node:  205       | \x1b[1mLeaf 0: 229\x1b[0m |       Node:  233         |        Node:  333                                           \n\x1b[1mLeaf 5: 1\x1b[0m | \x1b[1mLeaf 3: 1\x1b[0m |             | \x1b[1mLeaf 4: 94\x1b[0m | \x1b[1mLeaf 2: 139\x1b[0m | \x1b[1mLeaf 1: 146\x1b[0m |       Node:  187                              \n                                    |                          |             | \x1b[1mLeaf 6: 12\x1b[0m |                  Node:   18      \n                                    |                          |             |            |      Node:    3       | \x1b[1mLeaf 7: 4\x1b[0m\n                                    |                          |             |            | \x1b[1mLeaf 9: 1\x1b[0m | \x1b[1mLeaf 8: 1\x1b[0m |          \n"

assert model.__str__() == out_str
