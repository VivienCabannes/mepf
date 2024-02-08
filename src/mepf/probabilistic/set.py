"""
Batch Forever Elimination
"""
import heapq
from typing import List
import numpy as np
from ..binary_tree import EliminatedNode, Leaf, Tree


class BatchElimination(Tree):
    """
    Batch Elimination

    Attributes
    ----------
    partition: list of Vertex
        List of nodes in the current coarse partition (including the trash set).
    mode: Vertex
        Current empirical mode at partition level.
    nb_queries: int
        Total number of queries made.
    trash: EliminatedNode
        Node to store eliminated nodes.
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

        # elimination
        self.delta = 1 - confidence_level
        self.constant = constant

        # initialize leaves mappings
        y2leaf = [Leaf(value=0, label=i) for i in range(m)]
        self.trash = EliminatedNode()
        self.eliminated = np.zeros(m, dtype=bool)

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
        # if we have already eliminated all nodes, we do nothing
        if self.eliminated.sum() == self.m - 1:
            return

        codes = self.get_codes()
        self.root.value = len(y_cat)
        setattr(self, "y_codes", codes[y_cat])
        self._refine_partition(epsilon)
        delattr(self, "y_codes")

        if self.delta:
            sigma = np.log(((np.pi * self.root.value) ** 2) * self.m / self.delta)
            sigma = np.sqrt(self.constant * self.mode.value * sigma)
            criterion = self.mode.value - sigma
            tree_change = False
            remaining_node = []
            for node in self.partition:
                if isinstance(node, EliminatedNode):
                    continue
                if node.value < criterion:
                    self.trash.add_child(node)
                    tree_change = True
                else:
                    remaining_node.append(node)
            if tree_change:
                remaining_node.append(self.trash)
                self.partition = remaining_node
                self.trash_update()

        root = self.huffman_build(self.partition)
        self.replace_root(root)

    def _refine_partition(self, epsilon: float):
        """
        Find current :math:`\\eta`-admissible partition in the tree

        Here, :math:`\\eta = \\max N(y) / 2 n - \\epsilon`
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

            # we do not touch the eliminated set
            if isinstance(node, EliminatedNode):
                self.partition.append(node)
                continue

            # if have reached our stop criterion, we stop
            if n_mode is not None and node.value < n_mode / 2 - epsilon * n:
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

        # number of queries
        self.nb_queries += node.ind.sum()

    def get_leaves_set(self, node):
        """
        Get list of leaf descendants labels.
        """
        if type(node) is Leaf:
            y_set = [node.label]
        else:
            left_set = self.get_leaves_set(node.left)
            right_set = self.get_leaves_set(node.right)
            y_set = left_set + right_set
        return y_set

    def trash_update(self):
        """
        Update eliminated nodes.
        """
        y_set = []
        for child in self.trash.children:
            y_set += self.get_leaves_set(child)
        for y in y_set:
            self.eliminated[y] = True

    def __repr__(self):
        return f"BatchElimination at {id(self)}"


class SetElimination:
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
        return f"TruncatedBatchElimination at {id(self)}"

    def __str__(self):
        return self.back_end.__str__()
