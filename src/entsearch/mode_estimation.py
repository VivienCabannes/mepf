import logging
import numpy as np

log = logging.getLogger(__name__)


def mode_estimation(proba, T, get_maximal_partition, verbose=False):
    """
    Estimate the mode of a distribution with admissible splits.

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
    # Parse arguments and initialize variables
    m = len(proba)
    cdf = proba.cumsum()

    p_hat = np.ones(m) / m

    totals = {}
    counts = {}
    views = {}
    hits = {}

    all_ys = np.zeros(T, dtype=int)
    global_n_y = -1

    if verbose:
        estimated_prob = np.zeros(T)
        estimated_mode = np.zeros(T, dtype=int)
        old_t = 0

    t_T = t_1 = 0
    while t_T < T:
        t_1 += 1
        # Get maximal split given current probabilities estimates
        partition = get_maximal_partition(p_hat)
        log.debug("partition", partition)

        # Get hashable identifiers
        sets_id = {}
        for part_id in np.unique(partition):
            ind = partition == part_id
            set_id = ""
            for bit in ind:
                set_id += {True: "1,", False: "0,"}[bit]
            if set_id not in totals:
                totals[set_id] = 0
                counts[set_id] = 0
                views[set_id] = []
                hits[set_id] = []
            sets_id[part_id] = set_id

        for t_2 in range(t_1 + 1):
            # Draw a random variable Y_t and find which set it belongs to
            # log.debug(global_n_y, end=",", flush=True)
            global_n_y += 1
            Y_t = np.sum(np.random.rand() > cdf)
            all_ys[global_n_y] = Y_t
            answer = partition[Y_t]

            # Report statistics
            t_T += answer + 1
            if answer == np.max(partition):
                t_T -= 1  # Remove question whose answer could have been deduced
            for part_id in sets_id:
                set_id = sets_id[part_id]
                totals[set_id] += 1
                views[set_id].append(global_n_y)
                if answer == part_id:
                    counts[set_id] += 1
                    hits[set_id].append(global_n_y)
            if t_T >= T:
                break

            log.debug(f"q: {global_n_y}, t: {t_T}", flush=True)

        # Update probabilities estimate
        pS_hat = np.zeros(np.max(partition) + 1)
        for part_id in sets_id:
            set_id = sets_id[part_id]
            pS_hat[part_id] = counts[set_id] / totals[set_id]
        log.debug("pS_hat 1:", pS_hat)

        # If the best set is not a singleton, refine it.
        while np.sum(partition == np.argmax(pS_hat)) > 1:
            max_part_id = np.argmax(pS_hat)
            max_set_id = sets_id[max_part_id]
            # Refine partition
            p_hat[
                p_hat == 0
            ] = 1e-10  # Trick to avoid no split in get_greedy_maximal_partition
            sub_partition = get_maximal_partition(
                p_hat[partition == max_part_id]
            )
            sub_partition += np.max(partition) + 1
            partition[partition == max_part_id] = sub_partition
            log.debug("new_partition", partition)
            sub_ids = np.unique(sub_partition)

            # Update hashable identifiers
            for part_id in sub_ids:
                ind = partition == part_id
                set_id = ""
                for bit in ind:
                    set_id += {True: "1,", False: "0,"}[bit]
                if set_id not in totals:
                    totals[set_id] = 0
                    counts[set_id] = 0
                    views[set_id] = []
                    hits[set_id] = []
                sets_id[part_id] = set_id

            # Query the new sets
            for n_y in hits[max_set_id]:
                log.debug("recover value", n_y, all_ys[n_y])
                # If the sample is known to be in one of the set, we do not need additional queries
                already_hit = False
                for part_id in sub_ids:
                    if n_y in hits[sets_id[part_id]]:
                        already_hit = True
                if already_hit:
                    for part_id in sub_ids:
                        set_id = sets_id[part_id]
                        if n_y not in views[set_id]:
                            totals[set_id] += 1
                            views[set_id].append(n_y)
                else:
                    Y_t = all_ys[n_y]
                    answer = partition[Y_t]
                    assert answer in sub_ids

                    # Report statistics
                    t_T += answer + 1 - np.min(sub_ids)
                    if answer == np.max(sub_ids):
                        t_T -= 1  # Remove question whose answer could have been deduced
                    for part_id in sub_ids:
                        set_id = sets_id[part_id]
                        if n_y not in views[set_id]:
                            totals[set_id] += 1
                            views[set_id].append(n_y)
                            if answer == part_id:
                                counts[set_id] += 1
                                hits[set_id].append(n_y)
                        else:
                            if answer > part_id:
                                t_T -= 1  # Remove double counted questions
                            assert answer != part_id
                    log.debug(f"q: {global_n_y}, l_q: {n_y}, t: {t_T}", flush=True)
            # Update probabilities estimate
            pS_hat = np.zeros(np.max(partition) + 1)
            for part_id in np.unique(partition):
                set_id = sets_id[part_id]
                pS_hat[part_id] = counts[set_id] / totals[set_id]
            log.debug("pS_hat 2:", pS_hat)

        # Update the probabilities according to the answer
        for part_id in np.unique(partition):
            ind = partition == part_id
            p_hat[ind] = pS_hat[part_id] / np.sum(ind)

        if verbose:
            estimated_prob[old_t:t_T] = p_hat[np.argmax(proba)]
            estimated_mode[old_t:t_T] = np.argmax(p_hat)
            old_t = t_T

    if verbose:
        return estimated_mode, estimated_prob
    return np.argmax(p_hat)
