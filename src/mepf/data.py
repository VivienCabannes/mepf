import numpy as np


def sample_dirichlet(alpha, generator=np.random):
    """
    Sample from a Dirichlet distribution

    Parameters
    ----------
    alpha : array of shape (m,)
        Parameters of the distribution
    generator : numpy.random.Generator
        Random number generator

    Returns
    -------
    proba : array of shape (m,)
        Probability distribution
    """
    proba = generator.dirichlet(alpha)
    proba.sort()
    return proba[::-1]


def one_vs_all(m, p1):
    """
    Distribution with all but one elements equal

    Parameters
    ----------
    m : int
        Number of elements
    p1 : float
        Probability of the mode

    Returns
    -------
    proba : array of shape (m,)
        Probability distribution
    """
    assert p1 > 1 / m, "p1 is too small to be the mode of m elements"
    proba = np.zeros(m)
    proba[:] = (1 - p1) / (m - 1)
    proba[0] = p1
    return proba


def two_vs_all(m, p1, diff):
    """
    Distribution with two elements bigger than the rest

    Parameters
    ----------
    m : int
        Number of elements
    p1 : float
        Probability of the first element
    diff : float
        Difference between the first and second element

    Returns
    -------
    proba : array of shape (m,)
        Probability distribution
    """
    p2 = p1 - diff
    p3 = (1 - p1 - p2) / (m - 2)
    assert p1 > 1 / m, "p1 is too small to be the mode of m elements"
    assert p2 > p3, "p1 - diff is too small to be the second biggest element"
    assert p3 > 0, "1 - 2 * p1 + diff is negative"

    proba = np.zeros(m)
    proba[:] = p3
    proba[0] = p1
    proba[1] = p2
    return proba


def geometric(m, x):
    """
    Geometric progression distribution

    Parameters
    ----------
    m : int
        Number of samples
    x : float
        Common ratio of the geometric progression

    Returns
    -------
    proba : array of shape (m,)
        Probability distribution
    """
    proba = np.zeros(m)
    proba[:] = x
    proba **= -(1 + np.arange(m))
    proba /= proba.sum()
    return proba


def arithmetic(m, x):
    """
    Arithmetic progression distribution

    Parameters
    ----------
    m : int
        Number of elements
    x : float
        Unnormalzed common difference

    Returns
    -------
    proba : array of shape (m,)
        Probability distribution
    """
    proba = np.arange(m, dtype=float) + 1
    proba *= -x
    proba -= np.min(proba) - x
    proba /= proba.sum()
    return proba
