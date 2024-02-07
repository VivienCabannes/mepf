
import numpy as np

from mepf import (
    AdaptiveBatchSearch,
    BatchSearch,
    ExhaustiveSearch,
    TruncatedSearch,
)
from mepf.data import sample_dirichlet


def test_huffman_scheme():
    rng = np.random.default_rng(seed=1000)

    m = 10
    alpha = np.ones(m)
    proba = sample_dirichlet(alpha, generator=rng)

    n = 10_000
    y_cat = rng.choice(m, size=n, p=proba)

    model = ExhaustiveSearch(m, adaptive=True)
    for y in y_cat:
        model(y)

    # Check frequencies
    y_freqs = np.eye(m)[y_cat].sum(axis=0)
    for y in range(m):
        assert y_freqs[y] == model.y2leaf[y].value

    # Check Huffman ordering
    huffman_list = model.get_huffman_list()
    assert huffman_list == model.huffman_list
    old = huffman_list[0]
    for node in huffman_list[1:]:
        assert old < node
        old = node


def test_exhaustive_search(helpers):
    rng = np.random.default_rng(seed=1000)

    # generate data
    m = 10
    n = 1000
    alpha = np.ones(m)
    proba = sample_dirichlet(alpha, generator=rng)
    y_cat = rng.choice(m, size=n, p=proba)

    model = ExhaustiveSearch(m, adaptive=False)

    for y in y_cat:
        model(y)
    nb_queries_dichotomic = model.nb_queries

    helpers.reset_value(model)
    model.adaptive = True
    model.nb_queries = 0
    for y in y_cat:
        model(y)
    nb_queries_adaptive = model.nb_queries

    helpers.reset_value(model)
    model.adaptive = False
    model.nb_queries = 0
    for y in y_cat:
        model(y)
    nb_queries_huffman = model.nb_queries

    assert nb_queries_dichotomic == 3469
    assert nb_queries_adaptive == 3115
    assert nb_queries_huffman == 3085

    out_str = '                                                           Node: 1000                                                              \n                                             Node:  440         |                      Node:  560                                  \n      Node:  211                                  | \x1b[1mLeaf 0: 229\x1b[0m |        Node:  266         |        Node:  294                    \n\x1b[1mLeaf 4: 94\x1b[0m |                    Node:  117        |             | \x1b[1mLeaf 3: 127\x1b[0m | \x1b[1mLeaf 2: 139\x1b[0m | \x1b[1mLeaf 1: 146\x1b[0m |       Node:  148       \n           |       Node:   49        | \x1b[1mLeaf 7: 68\x1b[0m |             |                           |             | \x1b[1mLeaf 6: 70\x1b[0m | \x1b[1mLeaf 5: 78\x1b[0m\n           | \x1b[1mLeaf 8: 23\x1b[0m | \x1b[1mLeaf 9: 26\x1b[0m |            |             |                                                                  \n'

    assert model.__str__() == out_str


def test_batch_search(helpers):
    rng = np.random.default_rng(seed=0)

    m = 10
    n = 100_000
    alpha = np.ones(m)
    proba = sample_dirichlet(alpha, generator=rng)
    y_cat = rng.choice(m, size=n, p=proba)

    model = BatchSearch(m, adaptive=False)
    model_bis = BatchSearch(m, adaptive=True)

    for i in range(50):
        helpers.reset_value(model)
        y_cat = rng.choice(m, size=n, p=proba)

        model(y_cat, epsilon=0)

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
        model_bis(y_cat, epsilon=0)

    out_str = '        Node: 100000                                                                                                     \n\x1b[1mLeaf 0: 42826\x1b[0m |                                         Node:  57174                                                     \n              |                 Node:  21675                  |                                         Node:  35499     \n              |     Node:      0      |     Node:      0      |                 Node:      0                  | \x1b[1mLeaf 1: 0\x1b[0m\n              | \x1b[1mLeaf 4: 0\x1b[0m | \x1b[1mLeaf 8: 0\x1b[0m | \x1b[1mLeaf 6: 0\x1b[0m | \x1b[1mLeaf 2: 0\x1b[0m |     Node:      0      |     Node:      0      |          \n              |                                               | \x1b[1mLeaf 7: 0\x1b[0m | \x1b[1mLeaf 9: 0\x1b[0m | \x1b[1mLeaf 3: 0\x1b[0m | \x1b[1mLeaf 5: 0\x1b[0m |          \n'

    assert model.__str__() == out_str
    assert model_bis.__str__() == out_str


def test_truncated_search():
    rng = np.random.default_rng(seed=1000)

    m = 10
    n = 1_000
    alpha = np.ones(m)
    proba = sample_dirichlet(alpha, generator=rng)
    y_cat = rng.choice(m, size=n, p=proba)

    model = TruncatedSearch(m)
    finer_model = TruncatedSearch(m)

    for i, y in enumerate(y_cat):
        model(y, epsilon=0)
        finer_model(y, epsilon=1 / m)
        # finer_model.coarse_identification(y, epsilon=(i+1)**(-.5)/m)

        assert model.root.value == i + 1
        huffman_list = model.get_huffman_list()
        assert huffman_list[model._i_part:] == model.huffman_list[model._i_part:]
        old = huffman_list[model._i_part]
        for node in huffman_list[model._i_part + 1:]:
            assert not node < old
            old = node

    out_str = "                                 Node: 1000                                                                                   \n                   Node:  434         |                                 Node:  566                                            \n      Node:  205        | \x1b[1mLeaf 0: 229\x1b[0m |                   Node:  256         |        Node:  310                              \n\x1b[1mLeaf 5: 38\x1b[0m | \x1b[1mLeaf 3: 63\x1b[0m |             |      Node:  117        | \x1b[1mLeaf 2: 139\x1b[0m | \x1b[1mLeaf 1: 146\x1b[0m |      Node:  164                  \n                                      | \x1b[1mLeaf 8: 4\x1b[0m | \x1b[1mLeaf 4: 24\x1b[0m |             |             | \x1b[1mLeaf 7: 7\x1b[0m |      Node:   10      \n                                      |                                      |             |           | \x1b[1mLeaf 9: 1\x1b[0m | \x1b[1mLeaf 6: 1\x1b[0m\n"

    fin_str = "                                            Node: 1000                                                                        \n                              Node:  419         |                      Node:  581                                            \n      Node:  190                   | \x1b[1mLeaf 0: 229\x1b[0m |        Node:  266         |        Node:  315                              \n\x1b[1mLeaf 4: 94\x1b[0m |      Node:   96       |             | \x1b[1mLeaf 3: 127\x1b[0m | \x1b[1mLeaf 2: 139\x1b[0m | \x1b[1mLeaf 1: 146\x1b[0m |                  Node:  169      \n           | \x1b[1mLeaf 6: 1\x1b[0m | \x1b[1mLeaf 9: 1\x1b[0m |             |                           |             |      Node:    3       | \x1b[1mLeaf 7: 4\x1b[0m\n                                                 |                           |             | \x1b[1mLeaf 8: 1\x1b[0m | \x1b[1mLeaf 5: 2\x1b[0m |          \n"

    assert model.__str__() == out_str
    assert finer_model.__str__() == fin_str


def test_adaptive_batch_search(helpers):
    rng = np.random.default_rng(seed=0)

    m = 10
    n = 100_000
    alpha = np.ones(m)
    proba = sample_dirichlet(alpha, generator=rng)
    y_cat = rng.choice(m, size=n, p=proba)

    model = AdaptiveBatchSearch(m)
    model(y_cat)
    nb_queries = model.nb_queries

    assert model.back_end.root.value == len(y_cat) - 2 ** model.round + 2
    assert nb_queries == 157725

    out_str = '         Node: 34466                                                                                                     \n\x1b[1mLeaf 0: 14677\x1b[0m |                  Node: 19789                                                                             \n              |      Node:  6981      |                                          Node: 12808                             \n              | \x1b[1mLeaf 1: 0\x1b[0m | \x1b[1mLeaf 8: 0\x1b[0m |                  Node:    46                  |      Node:    54                 \n              |                       |      Node:     1      |      Node:     2      | \x1b[1mLeaf 2: 5\x1b[0m |      Node:     7     \n              |                       | \x1b[1mLeaf 4: 0\x1b[0m | \x1b[1mLeaf 6: 1\x1b[0m | \x1b[1mLeaf 7: 0\x1b[0m | \x1b[1mLeaf 9: 0\x1b[0m |           | \x1b[1mLeaf 3: 0\x1b[0m | \x1b[1mLeaf 5: 0\x1b[0m\n'

    assert model.__str__() == out_str


if __name__ == "__main__":
    from conftest import Helper
    helpers = Helper()

    test_huffman_scheme()
    test_exhaustive_search(helpers)
    test_batch_search(helpers)
    test_truncated_search()
    test_adaptive_batch_search(helpers)
