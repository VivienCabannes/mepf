import numpy as np

from entsearch.data import sample_dirichlet
from entsearch.probabilistic.elimination import ForeverElimination
from entsearch.probabilistic.set_elimination import ForeverSetElimination

rng = np.random.default_rng(seed=1000)


def test_forever_elimination():
    # generate data
    m = 10
    n = 1000
    alpha = np.zeros(m) + .1
    proba = sample_dirichlet(alpha, generator=rng)
    y_cat = rng.choice(m, size=n, p=proba)
    level = 0.5

    model = ForeverElimination(m, confidence_level=level, adaptive=False)
    for y in y_cat:
        model(y)
    nb_queries_dichotomic = model.nb_queries

    model = ForeverElimination(m, confidence_level=level, adaptive=True)
    for y in y_cat:
        model(y)
    nb_queries_adaptive = model.nb_queries

    model = ForeverSetElimination(m, confidence_level=level)
    for y in y_cat:
        model(y)
    nb_queries_set_adaptive = model.nb_queries

    assert nb_queries_dichotomic == 3261
    assert nb_queries_adaptive == 1426
    assert nb_queries_set_adaptive == 1007


if __name__ == '__main__':
    test_forever_elimination()
