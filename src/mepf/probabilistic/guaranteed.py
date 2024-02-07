"""
Guaranteed Set Search with batches
"""
from typing import List
from .batch import BatchElimination


class AdaptiveBatchElimination:
    """
    Adaptive Batch Elimination

    Attributes
    ----------
    back_end: BatchSearch
        Batch search back end to perform the search
    round: int
        Current round of the search
    """

    def __init__(self, m: int, confidence_level: float = 1, constant: float = 24):
        """
        Initialize the tree.

        Parameters
        ----------
        m:
            Maximal number of potential class
        confidence_level:
            Level of certitude require to eliminate guesses
        constant:
            Constant to resize confidence intervals, default is 24.
        """
        self.m = m
        self.back_end = BatchElimination(m, confidence_level=confidence_level, constant=constant)
        self.round = 0

    def get_scheduling(self, round):
        """
        Scheduling of batch size and admissibility
        """
        epsilon = (2 / 3) ** round / (4 * self.m)
        batch_size = 2 ** round
        return batch_size, epsilon

    def __call__(self, y_cat: List[int], epsilon: float = 0):
        """
        Find the emprical mode in a batch

        Parameters
        ----------
        y_cat:
            Batch observation
        epsilon:
            Criterion on the biggest `p(S)` compared to `max p(y)`,
            for any non-singleton set :math:`p(S) < \\max_y p(y) - \\epsilon`
        """
        start_index = 0
        while start_index < len(y_cat):
            self.round += 1
            batch_size, epsilon = self.get_scheduling(self.round)
            end_index = start_index + batch_size
            y_cur = y_cat[start_index:end_index]
            self.back_end(y_cur, epsilon)
            start_index = end_index

    @property
    def mode(self):
        return self.back_end.mode

    @property
    def nb_queries(self):
        return self.back_end.nb_queries

    def __repr__(self):
        return f"TruncatedBatchSearch at {id(self)}"

    def __str__(self):
        return self.back_end.__str__()
