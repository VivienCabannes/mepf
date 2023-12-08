import numpy as np

from entsearch import SearchTree
from entsearch.data import sample_dirichlet

rng = np.random.default_rng(seed=0)

m = 10
alpha = np.ones(m)
proba = sample_dirichlet(alpha, generator=rng)

n = 100_000
y_cat = rng.choice(m, size=n, p=proba)
y_one_hot = np.eye(m)[y_cat]
freqs = y_one_hot.sum(axis=0)

model = SearchTree([1 for _ in range(m)])

for i in range(50):
    model.reset_value()
    y_cat = rng.choice(m, size=n, p=proba)
    # observations = np.ones((n, m), dtype=bool)

    partition = model.find_admissible_partition(y_cat, epsilon=0)

    nb_queries = 0
    for node in partition:
        nb_queries += node.depth * node.value

    nb_queries_bis = -n
    node_list = [model.root]
    while len(node_list) > 0:
        node = node_list.pop(0)
        if node.value is None:
            continue
        nb_queries_bis += node.value
        if not hasattr(node, "left"):
            continue
        if node.left is not None:
            node_list.append(node.left)
            node_list.append(node.right)

    assert nb_queries_bis == nb_queries

    node = model.huffman_build(partition)
    model.replace_root(node)
    model.codes = model.get_codes()

out_str = '        Node: 100000                                                                                                                                \n\x1b[1mLeaf 0: 42826\x1b[0m |                                                     Node:  57174                                                                    \n              |                       Node:  21675                        |                                                     Node:  35499        \n              |         Node: None          |         Node: None          |                        Node: None                         | \x1b[1mLeaf 1: None\x1b[0m\n              | \x1b[1mLeaf 4: None\x1b[0m | \x1b[1mLeaf 8: None\x1b[0m | \x1b[1mLeaf 6: None\x1b[0m | \x1b[1mLeaf 2: None\x1b[0m |         Node: None          |         Node: None          |             \n              |                                                           | \x1b[1mLeaf 7: None\x1b[0m | \x1b[1mLeaf 9: None\x1b[0m | \x1b[1mLeaf 3: None\x1b[0m | \x1b[1mLeaf 5: None\x1b[0m |             \n'

assert model.__str__() == out_str
