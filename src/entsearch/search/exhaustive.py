
import numpy as np
from ..huffman import huffman_codes


def dichotomic_baseline(Y):
    """
    Perform dichotomic search on the given sequence.

    Parameters
    ----------
    Y : numpy.ndarray
        Sequence of observations.

    Returns
    -------
    T : numpy.ndarray
        Array of the number of queries.
    mode : numpy.ndarray
        Array of the estimated modes.
    p_mode : numpy.ndarray
        Array of the estimated probabilities of the modes.
    """
    n, m = Y.shape
    S = huffman_codes([1 for i in range(m)])
    T = (Y @ S != -1).sum(axis=1).cumsum()
    p_hat = Y.cumsum(axis=0)
    p_hat /= np.arange(1, n + 1)[:, np.newaxis]
    mode = np.argmax(p_hat, axis=1)
    p_mode = p_hat[np.arange(n), mode]
    return T, mode, p_mode
