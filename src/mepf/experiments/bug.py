import numpy as np

m = 100
n_data = 5
proba = np.zeros(m)
proba[:] = 1 / (2 * (m - 1))
proba[0] = 1 / 2
proba /= proba.sum()

num_exp = 100_000
count = 0
for _ in range(num_exp):
    data = np.random.choice(m, size=n_data, p=proba)
    cla, fre = np.unique(data, return_counts=True)
    if (np.max(fre) == fre).sum() == 1 and cla[np.argmax(fre)] == 0:
        # if cla[np.argmax(fre)] == 0:
        count += 1
print("real delta", 1 - count / num_exp)

sanov = np.exp(n_data * (np.log(1 - (np.sqrt(1 / 2) - np.sqrt(proba[1])) ** 2)))
print("predicted delta", sanov)
