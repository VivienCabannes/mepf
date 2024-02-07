import numpy as np

from entsearch.data import sample_dirichlet
from entsearch.probabilistic.elimination import Elimination
from entsearch.probabilistic.set import SetElimination
from entsearch.probabilistic.batch import BatchElimination

rng = np.random.default_rng(seed=1000)


def test_forever_elimination(helpers):
    # generate data
    m = 10
    n = 1000
    alpha = np.zeros(m) + .1
    proba = sample_dirichlet(alpha, generator=rng)
    y_cat = rng.choice(m, size=n, p=proba)
    level = 0.5

    model = Elimination(m, confidence_level=level, adaptive=False)
    for y in y_cat:
        model(y)
    nb_queries_dichotomic = model.nb_queries

    model = Elimination(m, confidence_level=level, adaptive=True)
    for y in y_cat:
        model(y)
    nb_queries_adaptive = model.nb_queries

    model = SetElimination(m, confidence_level=level)
    for y in y_cat:
        model(y)
    nb_queries_set = model.nb_queries

    model = BatchElimination(m, confidence_level=level)
    model(y_cat)
    nb_queries_batch = model.nb_queries
    for i in range(10):
        helpers.reset_value(model)
        model.nb_queries = 0
        model(y_cat)
    nb_queries_best_batch = model.nb_queries

    assert nb_queries_dichotomic == 2555
    assert nb_queries_adaptive == 1198
    assert nb_queries_set == 1007
    assert nb_queries_batch == 3666
    assert nb_queries_best_batch == 1000


if __name__ == '__main__':
    from conftest import Helper
    helpers = Helper()
    test_forever_elimination(helpers)
