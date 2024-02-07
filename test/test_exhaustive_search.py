
import numpy as np

from mepf import ExhaustiveSearch
from mepf.data import sample_dirichlet


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


if __name__ == "__main__":
    from conftest import Helper
    helpers = Helper()
    test_exhaustive_search(helpers)
