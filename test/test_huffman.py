
import numpy as np

from entsearch.search import HuffmanTree
from entsearch.data import sample_dirichlet

rng = np.random.default_rng(seed=1000)

m = 10
alpha = np.ones(m)
proba = sample_dirichlet(alpha, generator=rng)

codes = HuffmanTree(proba).get_codes()
dichotomic_codes = HuffmanTree([1 for _ in range(m)]).get_codes()

n = 1000
y_cat = rng.choice(m, size=n, p=proba)
y_one_hot = np.zeros((n, m))
y_one_hot[np.arange(n), y_cat] = 1

nb_queries_huffman = (y_one_hot @ codes != -1).sum(axis=1)
nb_queries_dichotomic = (y_one_hot @ dichotomic_codes != -1).sum(axis=1)

assert nb_queries_dichotomic.sum() == 3469
assert nb_queries_huffman.sum() == 3085

model = HuffmanTree([0 for _ in range(m)])
nb_queries_adaptative = np.zeros(n, dtype=int)
n = len(y_cat)
for i in range(n):
    y = y_cat[i]
    nb_queries_adaptative[i] = model.y2leaf[y].depth
    model.report_observation(y)

out_str = '                                                           Node: 1000                                                              \n                                             Node:  440         |                      Node:  560                                  \n      Node:  211                                  | \x1b[1mLeaf 0: 229\x1b[0m |        Node:  266         |        Node:  294                    \n\x1b[1mLeaf 4: 94\x1b[0m |                    Node:  117        |             | \x1b[1mLeaf 3: 127\x1b[0m | \x1b[1mLeaf 2: 139\x1b[0m | \x1b[1mLeaf 1: 146\x1b[0m |       Node:  148       \n           |       Node:   49        | \x1b[1mLeaf 7: 68\x1b[0m |             |                           |             | \x1b[1mLeaf 6: 70\x1b[0m | \x1b[1mLeaf 5: 78\x1b[0m\n           | \x1b[1mLeaf 8: 23\x1b[0m | \x1b[1mLeaf 9: 26\x1b[0m |            |             |                                                                  \n'

assert model.__str__() == out_str
