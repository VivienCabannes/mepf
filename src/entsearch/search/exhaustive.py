import numpy as np


def get_nb_queries(y_cat, codes):
    """
    Perform dichotomic search on the given sequence.

    Parameters
    ----------
    y_cat : numpy.ndarray of int
        Sequence of observations.
    codes: numpy.ndarray of int
        Codes of each symbol.

    Returns
    -------
    nb_queries : numpy.ndarray
        Number of queries make to identify each y.
    """
    n = len(y_cat)
    m = len(codes)
    y_one_hot = np.zeros((n, m))
    y_one_hot[np.arange(n), y_cat] = 1
    nb_queries = (y_one_hot @ codes != -1).sum(axis=1)
    return nb_queries


def get_exhaustive_statistics(y_cat, m):
    """
    Compute sequences of empirical modes and their empirical frequencies.

    Parameters
    ----------
    y_cat : numpy.ndarray of int
        Sequence of observations.
    m : int
        Number of symbols.

    Returns
    -------
    hat_mode : numpy.ndarray of int
        Sequence of empirical modes.
    hat_p_mode : numpy.ndarray of float
        Sequence of empirical frequencies.
    """
    n = len(y_cat)
    p_hat = np.zeros((n, m))
    p_hat[np.arange(n), y_cat] = 1
    np.cumsum(p_hat, out=p_hat, axis=0)
    p_hat /= np.arange(1, n + 1)[:, np.newaxis]
    hat_mode = np.argmax(p_hat, axis=1)
    hat_p_mode = p_hat[np.arange(n), hat_mode]
    return hat_mode, hat_p_mode
