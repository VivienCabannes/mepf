"""
Truncated Tree Search
"""
import heapq
from typing import List
import numpy as np
from ..binary_tree import Leaf, Tree


class BatchSearch(Tree):
    """
    Batch Tree Search

    Attributes
    ----------
    partition: list of Vertex
        List of nodes in the current coarse partition.
    mode: Vertex
        Current empirical mode at partition level.
    nb_queries: int
        Total number of queries made.
    """

    def __init__(self, m: int, adaptive: bool = True):
        """
        Initialize the tree.

        Parameters
        ----------
        m:
            Maximal number of potential class
        adaptive:
            Wether to update the tree online
        """
        self.m = m
        self.adaptive = adaptive

        y2leaf = [Leaf(value=0, label=i) for i in range(m)]

        # initialize the tree
        root = Tree.build_balanced_subtree(y2leaf)
        Tree.__init__(self, root)

        # initialize the partition
        self.partition = [root]

        # initialize mode guess
        self.mode = y2leaf[0]
        self.nb_queries = 0

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
        codes = self.get_codes()
        self.root.value = len(y_cat)
        setattr(self, "y_codes", codes[y_cat])
        self._refine_partition(epsilon)
        delattr(self, "y_codes")

        if self.adaptive:
            root = self.huffman_build(self.partition)
            self.replace_root(root)

    def _refine_partition(self, epsilon: float):
        """
        Find current :math:`\\eta`-admissible partition in the tree

        Here, :math:`\\eta = \\max N(y) / n - \\epsilon`
        """
        # we are splitting nodes
        self.root.ind = np.ones(self.root.value, dtype=bool)
        n_mode = None
        n = self.root.value

        # update the partition
        heap = []
        heapq.heappush(heap, (-self.root.value, self.root))
        self.partition = []
        while len(heap) > 0:
            # get the node with the biggest value
            _, node = heapq.heappop(heap)

            # if have reached our stop criterion, we stop
            if n_mode is not None and node.value < n_mode - epsilon * n:
                self.partition.append(node)
                break

            # the first leaf we find if the mode
            if isinstance(node, Leaf):
                self.partition.append(node)
                if n_mode is None:
                    self.mode = node
                    n_mode = node.value
                continue

            # make n_node queries to get children information
            self._report_count(node, self.y_codes)

            # push children in the heap
            # be careful that we need to inverse the order
            loffset = {False: 0, True: 0.5}[type(node.left) is Leaf]
            roffset = {False: 0, True: 0.5}[type(node.right) is Leaf]
            heapq.heappush(heap, (-node.left.value + loffset, node.left))
            heapq.heappush(heap, (-node.right.value + roffset, node.right))

        # the remaning node form the partition
        while len(heap) > 0:
            _, node = heapq.heappop(heap)
            self.partition.append(node)

    def _report_count(self, node, y_codes):
        """
        Query observation to refine information at node level

        Parameters
        ----------
        node: Node
            Node to refine observations
        y_codes: numpy.ndarray of int of size (n, c)
            Matrix of codes for each observed class
        """
        if node.value == 0:
            return

        # get child dispenser
        pos_ind = y_codes[node.ind, node.depth] == 1

        node.right.ind = node.ind.copy()
        node.right.ind[node.ind] = pos_ind
        node.right.value = np.sum(pos_ind)

        node.left.ind = node.ind.copy()
        node.left.ind[node.ind] = ~pos_ind
        node.left.value = node.value - node.right.value

        # number of queries if we do not remember past information
        self.nb_queries += node.ind.sum()

    def __repr__(self):
        return f"BatchSearch at {id(self)}"


class TruncatedSearch:
    """
    Adaptive Batch Search

    Attributes
    ----------
    back_end: BatchSearch
        Batch search back end to perform the search
    round: int
        Current round of the search
    """

    def __init__(self, m: int):
        """
        Initialize the tree.

        Parameters
        ----------
        m:
            Maximal number of potential class
        adaptive:
            Wether to update the tree online
        """
        self.m = m
        self.back_end = BatchSearch(m, adaptive=True)
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

