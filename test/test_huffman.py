import numpy as np

from entsearch.search import SearchTree
from entsearch.data import sample_dirichlet


def test_huffman():
    rng = np.random.default_rng(seed=1000)

    m = 10
    alpha = np.ones(m)
    proba = sample_dirichlet(alpha, generator=rng)

    n = 10_000
    y_cat = rng.choice(m, size=n, p=proba)

    model = SearchTree(m)
    for y in y_cat:
        model.fine_identification(y, update=True)

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


if __name__ == "__main__":
    test_huffman()
