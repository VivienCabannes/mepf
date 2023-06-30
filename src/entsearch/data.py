
import numpy as np


def proba_generator(m, epsilon):
    """
    Generate a probability vector of size `m` through the formula
     
    .. math::
        p(y) \propto \exp(`epsilon` * x), x\sim N(0, 1)
    
    Parameters
    ----------
    m : int
        Size of the probability vector.
    epsilon : float
        Parameter of the probability vector.

    Returns
    -------
    proba : np.ndarray
        Probability vector of size `m`.
    """

    proba = np.exp(epsilon * np.random.randn(m))
    proba /= proba.sum()
    return proba
