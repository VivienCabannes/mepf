import numpy as np

from mepf import (
    nb_data_required,
    ExhaustiveSearch,
    RoundFreeTruncatedSearch,
    TruncatedSearch,
    BatchElimination,
    Elimination,
    RoundFreeSetElimination,
)
from mepf.data import (
    sample_dirichlet,
    geometric,
    arithmetic,
    one_vs_all,
    two_vs_all,
)


rng = np.random.default_rng(seed=1000)

# generate data
m = 6
alpha = np.zeros(m) + .1
# proba = sample_dirichlet(alpha, generator=rng)
# proba = one_vs_all(m, p1=1 / 2)
# proba = two_vs_all(m, p1=5 / m, diff=.01)
proba = arithmetic(m)
# proba = geometric(m)
proba = rng.permutation(proba)

confidence_level = 1 - 10 ** -2
nb_queries = {}
guesses = {'truth': np.argmax(proba)}

n = nb_data_required(proba, confidence_level)
y_cat = rng.choice(m, size=n, p=proba)
print(n)

model = ExhaustiveSearch(m)
for y in y_cat:
    model(y)
nb_queries['exhaustive'] = model.nb_queries
guesses['exhaustive'] = model.mode.label

model = ExhaustiveSearch(m, adaptive=True)
for y in y_cat:
    model(y)
nb_queries['adaptive_exhaustive'] = model.nb_queries
guesses['adaptive_exhaustive'] = model.mode.label

model = TruncatedSearch(m)
model(y_cat)
nb_queries['truncated_search'] = model.nb_queries
guesses['truncated_search'] = model.mode.label

model = RoundFreeTruncatedSearch(m)
for i, y in enumerate(y_cat):
    model(y, epsilon=(2 / 3) ** np.log2(i) / (4 * m))
nb_queries['heuristic_truncated_search'] = model.nb_queries
guesses['heuristic_truncated_search'] = model.mode.label

constant = 24
confidence_level = 1 - 10 ** -1

model = Elimination(m, confidence_level=confidence_level, constant=constant)
i = 0
while model.eliminated.sum() < m - 1 and i < n:
    i += 1
    model(rng.choice(m, p=proba))
nb_queries['elimination'] = model.nb_queries
guesses['elimination'] = model.mode.label

constant = 24
error = 0
for i in range(100):
    model = BatchElimination(m, confidence_level=confidence_level, constant=constant)
    r = 0
    while model.eliminated.sum() < m - 1:
        r += 1
        epsilon = (2 / 3) ** r / (4 * m)
        y_cat = rng.choice(m, size=2 ** r, p=proba)
        model(y_cat, epsilon=epsilon)
    nb_queries['set_elimination'] = model.nb_queries
    guesses['set_elimination'] = model.mode.label
    error += int(model.mode.label != np.argmax(proba))

print('error', error)

model = RoundFreeSetElimination(m, confidence_level=confidence_level, constant=constant, gamma=.5)
i = 0
while model.eliminated.sum() < m - 1:
    i += 1
    if i == 49:
        pass
    model(rng.choice(m, p=proba))
print(model)
nb_queries['round_free_set_elimination'] = model.nb_queries
guesses['round_free_set_elimination'] = model.mode.label

for key in nb_queries:
    print(f"{key}: {nb_queries[key]}")

print()
for key in guesses:
    print(f"{key}: {guesses[key]}")
