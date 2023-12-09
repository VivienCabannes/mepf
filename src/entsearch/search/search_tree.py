"""
Search Tree
"""
import heapq
import numpy as np
from .tree_constructors import Tree, Leaf


class SearchTree(Tree):
    """
    Search Tree implementation

    Attributes
    ----------
    root: Node
        The root of the search tree.
    y2leaf: dict of int: Leaf
        Dictionary mapping each class to its corresponding leaf.
    codes: list of list of int
        List of all leaf codes according to their position in the tree.
    """

    def __init__(self, counts):
        """
        Initialize the search tree.

        Parameters
        ----------
        codes : list of list of int or float
            Counts of each class to initialize the tree.
        """
        # initialize the leaves
        self.m = len(counts)
        nodes = [Leaf(value=counts[i], label=i) for i in range(self.m)]
        self.y2leaf = {i: nodes[i] for i in range(self.m)}

        # build the Huffman tree from counts
        root = self.huffman_build(nodes)
        Tree.__init__(self, root)
        self.codes = self.get_codes()

    @staticmethod
    def report_count(node, y_codes):
        """
        Query observation to refine information at node level

        Parameters
        ----------
        node: Node
            Node to refine observations
        y_codes: numpy.ndarray of int of size (n, c)
            Matrix of codes for each observed class
        """
        assert node.value != 0, "Node has no observation reported"

        # get child dispenser
        pos_ind = y_codes[node.ind, node.depth] == 1

        node.right.ind = node.ind.copy()
        node.right.ind[node.ind] = pos_ind
        node.right.value = np.sum(pos_ind)

        node.left.ind = node.ind.copy()
        node.left.ind[node.ind] = ~pos_ind
        node.left.value = node.value - node.right.value

    def find_admissible_partition(self, y_cat, epsilon=0):
        """
        Number of queries need to find the empirical mode in a sequence

        Parameters
        ----------
        y_cat: list of int of size (n,)
            Sequential samples of the categorical variable
        epsilon: float, optional
            Stopping criterion for the difference in probability between the
            empirical mode and the sets in the found partition

        Returns
        -------
        partition: list
            List of nodes in the `epsilon`-admissible partition

        Notes
        -----
        This function sets tree nodes values to number of observation
        """
        y_codes = self.codes[y_cat]
        n = len(y_codes)
        self.root.ind = np.ones(n, dtype=bool)
        self.root.value = n

        # initialize the heap
        heap = []
        heapq.heappush(heap, (-n, self.root))

        partition = []
        n_mode = None
        while len(heap) > 0:
            # get the node with the biggest value
            _, node = heapq.heappop(heap)
            # if the node value is smaller than our stop criterion, we stop
            if n_mode is not None and node.value <= n_mode - epsilon * n:
                partition.append(node)
                break
            # special behavior for leaves, since they have no children
            if isinstance(node, Leaf):
                partition.append(node)
                # if it is the first leaf, we found the empirical mode
                if n_mode is None:
                    n_mode = node.value
                continue
            # make n_node queries to get children information
            self.report_count(node, y_codes)
            # push children in the heap
            heapq.heappush(heap, (-node.left.value, node.left))
            heapq.heappush(heap, (-node.right.value, node.right))
        while len(heap) > 0:
            _, node = heapq.heappop(heap)
            partition.append(node)
        return partition

    def process_batch(self, y_cat, epsilon=0, adapt=True):
        """
        Process a batch of observations to update the tree

        Parameters
        ----------
        y_cat: list of int
            Sequential samples of the categorical variable
        epsilon: float, optional
            Stopping criterion for the difference in probability between the
            empirical mode and the sets in the found partition
        adapt: bool, optional
            Whether to adapt the tree to the partition found

        Returns
        -------
        nb_queries: int
            Number of queries to identify the partition
        """
        partition = self.find_admissible_partition(y_cat, epsilon=epsilon)
        if adapt:
            root = self.huffman_build(partition)
            self.replace_root(root)
            self.codes = self.get_codes()
        nb_queries = 0
        for node in partition:
            nb_queries += node.value * node.depth
        return nb_queries

    def get_nb_queries(self, y_cat):
        return self.process_batch(y_cat, epsilon=0, adapt=False)

    def get_nb_queries_sequential(self, y_cat):
        """
        Get the number of queries made to identify the empirical mode
        with sequential search

        Parameters
        ----------
        y_cat: list of int
            Sequential samples of the categorical variable

        Returns
        -------
        nb_queries: int
            Number of queries to identify the empirical mode
        """
        n = len(y_cat)
        m = len(self.codes)
        y_one_hot = np.zeros((n, m))
        y_one_hot[np.arange(n), y_cat] = 1
        nb_queries = (y_one_hot @ self.codes != -1).sum()
        return nb_queries

    def __repr__(self):
        return f"SearchTree at {id(self)}"
