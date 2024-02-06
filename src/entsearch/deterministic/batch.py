"""
Batch Tree Search
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
    y_cat: np.ndarray
        List of past samples.
    y_observations: np.ndarray
        List of weak observation of past samples.
    """

    def __init__(self, m: int, comeback: bool = False, adaptive: bool = True):
        """
        Initialize the tree.

        Parameters
        ----------
        m:
            Maximal number of potential class
        comeback:
            Wether to remember past information for future re-query
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

        if comeback:
            # remember past information for comeback
            self.y_cat = np.arange(m, dtype=int)
            self.y_observations = np.eye(m, dtype=bool)[np.arange(m)]

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
        self.get_leaves_set(self.root)
        self._refine_partition(epsilon)
        delattr(self, "y_codes")

        if self.adaptive:
            root = self.huffman_build(self.partition)
            self.replace_root(root)

    def _refine_partition(self, epsilon: float):
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

        rcode, lcode = node.right._set_code, node.left._set_code

        # report nb_queries
        if hasattr(self, "y_observations"):
            # only queries for new information
            y_obs = self.y_observations[: self.root.value]
            right_unknown = y_obs[node.right.ind] & ~rcode
            right_queries = (right_unknown.sum(axis=1) != 0).sum()
            left_unknown = y_obs[node.left.ind] & ~lcode
            left_queries = (left_unknown.sum(axis=1) != 0).sum()
            self.nb_queries += right_queries + left_queries
            # update observations
            self.y_observations[: self.root.value][node.right.ind] &= rcode
            self.y_observations[: self.root.value][node.left.ind] &= lcode
        else:
            # number of queries if we do not remember past information
            self.nb_queries += node.ind.sum()

    def __repr__(self):
        return f"BatchSearch at {id(self)}"

    def get_leaves_set(self, node):
        """
        Get list of leaf descendants labels and set codes.
        """
        if type(node) is Leaf:
            y_set = [node.label]
        else:
            left_set = self.get_leaves_set(node.left)
            right_set = self.get_leaves_set(node.right)
            y_set = left_set + right_set
        node._set_code = np.zeros(self.m, dtype=bool)
        node._set_code[y_set] = 1
        return y_set
