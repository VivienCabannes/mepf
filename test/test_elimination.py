import numpy as np

from entsearch.data import sample_dirichlet
from entsearch.probabilistic.forever_elimination import ForeverElimination

rng = np.random.default_rng(seed=1000)

# generate data
m = 10
n = 1000
alpha = np.zeros(m) + .1
proba = sample_dirichlet(alpha, generator=rng)
y_cat = rng.choice(m, size=n, p=proba)

model = ForeverElimination(m, confidence_level=.5, adaptive=False)
for y in y_cat:
    model(y)
nb_queries_dichotomic = model.nb_queries

model = ForeverElimination(m, confidence_level=.5, adaptive=True)
for y in y_cat:
    model(y)
nb_queries_adaptive = model.nb_queries

assert nb_queries_dichotomic == 3261
assert nb_queries_adaptive == 1426
