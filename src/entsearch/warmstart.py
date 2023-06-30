import numpy as np


def bias_mode_estimation(proba: np.ndarray, T: int, verbose: bool = False):
    """
    Estimate the mode of a distribution with a biased algorithm.

    Parameters
    ----------
    proba : array-like
        The probability distribution to estimate the mode of.
    T : int
        The number of iterations of the algorithm.
    verbose : bool, optional
        Whether to return intermediate results of the algorithm.

    Returns
    -------
    mode: int
        The estimated mode of the distribution.
    """
    # Parse arguments
    m = len(proba)
    cdf = proba.cumsum()
    p_hat = np.ones(m) / m

    if verbose:
        estimated_prob = np.zeros(T)
        estimated_mode = np.zeros(T, dtype=int)

    for t in range(T):
        # Choose a random permutation of the indices to build a maximal subset
        sigma = np.random.permutation(m)
        S_t = np.zeros(m, dtype=bool)
        S_sum = 0
        p_max = np.max(p_hat)

        for i in range(m):
            p_hat_y = p_hat[sigma[i]]
            if S_sum + p_hat_y <= p_max:
                S_t[sigma[i]] = True
                S_sum += p_hat_y

        # Draw a random variable Y_t and ask if it belongs to S_t
        Y_t = np.sum(np.random.rand() > cdf)
        answer = S_t[Y_t]

        # Update the probabilities according to the answer
        p_hat *= t / (t + 1)
        if answer:
            p_hat[S_t] += 1 / (np.sum(S_t) * (t + 1))
        else:
            p_hat[~S_t] += 1 / (np.sum(~S_t) * (t + 1))

        if verbose:
            estimated_prob[t] = p_hat[np.argmax(proba)]
            estimated_mode[t] = np.argmax(p_hat)

    if verbose:
        return estimated_mode, estimated_prob
    return np.argmax(p_hat)


def unbias_mode_estimation(proba, T, get_maximal_partition, verbose=False):
    """
    Estimate the mode of a distribution with an unbiased algorithm.

    Parameters
    ----------
    proba : array-like
        The probability distribution to estimate the mode of.
    T : int
        The number of iterations of the algorithm.
    get_maximal_partition : callable
        A function that returns a maximal partition of the probability
    verbose : bool, optional
        Whether to return intermediate results of the algorithm.

    Returns
    -------
    mode: int
        The estimated mode of the distribution.
    """
    # Parse arguments
    m = len(proba)
    cdf = proba.cumsum()
    p_hat = np.ones(m) / m

    if verbose:
        estimated_prob = np.zeros(T)
        estimated_mode = np.zeros(T, dtype=int)
        old_t = 0

    t_T = t_p = t_1 = 0
    while t_T < T:
        t_1 += 1
        # Get a maximal partition
        partition = get_maximal_partition(p_hat)

        for t_2 in range(t_1):
            # Draw a random variable Y_t and find which set it belongs to
            Y_t = np.sum(np.random.rand() > cdf)
            answer = partition[Y_t]

            # Report number of asked questions to get the positive answer
            t_T += answer + 1
            if t_T >= T:
                if verbose:
                    estimated_prob[old_t:t_T] = p_hat[np.argmax(proba)]
                    estimated_mode[old_t:t_T] = np.argmax(p_hat)
                break

            # Update the probabilities according to the answer
            p_hat *= t_p / (t_p + 1)
            p_hat[partition == answer] += 1 / (np.sum(partition == answer) * (t_p + 1))
            t_p += 1

            if verbose:
                estimated_prob[old_t:t_T] = p_hat[np.argmax(proba)]
                estimated_mode[old_t:t_T] = np.argmax(p_hat)
                old_t = t_T

    if verbose:
        return estimated_mode, estimated_prob
    return np.argmax(p_hat)
