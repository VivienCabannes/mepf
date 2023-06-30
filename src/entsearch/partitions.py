import numpy as np


def get_greedy_maximal_partition(proba):
    """
    Get a maximal partition using a greedy algorithm.

    Parameters
    ----------
    proba : array-like
        The probability distribution used to defined maximal partitions.

    Returns
    -------
    partition: array-like
        An approximate maximal partition associated the probability distribution.
    """
    m = len(proba)
    p_max = np.max(proba)
    partition = np.zeros(m, dtype=int)
    sigma = np.random.permutation(m)
    counter = 0
    while (partition == 0).any():
        counter += 1
        S_sum = 0
        for i in range(m):
            if not partition[sigma[i]] and S_sum + proba[sigma[i]] <= p_max:
                partition[sigma[i]] = counter
                S_sum += proba[sigma[i]]
    partition -= 1
    return partition


def get_singleton_partition(proba):
    """
    Get partition of the set of elemetns with singletons for baseline purposes.

    Parameters
    ----------
    proba : array-like
        The probability distribution used to defined maximal partitions.

    Returns
    -------
    partition: array-like
        The partition into singleton.
    """
    return np.arange(len(proba))
