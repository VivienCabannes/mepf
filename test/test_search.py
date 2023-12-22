import numpy as np

from entsearch.search import SearchTree
from entsearch.data import sample_dirichlet


def test_search(helpers):
    rng = np.random.default_rng(seed=0)

    m = 10
    n = 100_000
    alpha = np.ones(m)
    proba = sample_dirichlet(alpha, generator=rng)
    y_cat = rng.choice(m, size=n, p=proba)

    model = SearchTree(m)
    model_bis = SearchTree(m)

    for i in range(50):
        helpers.reset_value(model)
        y_cat = rng.choice(m, size=n, p=proba)

        model.batch_identification(y_cat, epsilon=0, update=False)

        nb_queries = 0
        for node in model.partition:
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

        root = model.huffman_build(model.partition)
        model.replace_root(root)
        model.huffman_list = model.get_huffman_list()
        for i, node in enumerate(model.huffman_list):
            node._i_huff = i

        helpers.reset_value(model_bis)
        model_bis.batch_identification(y_cat, epsilon=0, update=True)

    out_str = '        Node: 100000                                                                                                     \n\x1b[1mLeaf 0: 42826\x1b[0m |                                         Node:  57174                                                     \n              |                 Node:  21675                  |                                         Node:  35499     \n              |     Node:      0      |     Node:      0      |                 Node:      0                  | \x1b[1mLeaf 1: 0\x1b[0m\n              | \x1b[1mLeaf 4: 0\x1b[0m | \x1b[1mLeaf 8: 0\x1b[0m | \x1b[1mLeaf 6: 0\x1b[0m | \x1b[1mLeaf 2: 0\x1b[0m |     Node:      0      |     Node:      0      |          \n              |                                               | \x1b[1mLeaf 7: 0\x1b[0m | \x1b[1mLeaf 9: 0\x1b[0m | \x1b[1mLeaf 3: 0\x1b[0m | \x1b[1mLeaf 5: 0\x1b[0m |          \n'

    assert model.__str__() == out_str
    assert model_bis.__str__() == out_str


if __name__ == "__main__":
    from conftest import Helper
    helpers = Helper()
    test_search(helpers)
