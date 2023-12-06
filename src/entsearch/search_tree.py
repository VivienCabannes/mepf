"""
Search Tree
"""
import heapq
import numpy as np
from .tree_constructors import Tree, Node, Leaf


class SearchTree(Tree):
    """
    Search Tree implementation

    Attributes
    ----------
    root: Node
        The root of the Search tree.
    y2leaf: dict of int: Leaf
        Dictionary mapping each class to its corresponding leaf.
    codes: list of list of int
        List of all leaf codes according to their position in the tree.
    """

    def __init__(self, codes):
        """
        Initialize the Search tree with integer codes.

        Parameters
        ----------
        codes : list of list of int
            The codes of each class to initialize the tree.
        """
        Tree.__init__(self, Node())
        self.codes = codes
        self.y2leaf = {}

        # to build the search tree, iterate over the classess
        for y in range(len(codes)):
            code = codes[y]
            self.y2leaf[y] = Leaf(0, y)
            current = self.root
            # iterate over the class code
            for i in range(len(code)):

                # if c_i=1, we go down to the right child
                if code[i] == 1:
                    if isinstance(current.right, Leaf):
                        assert (
                            current.right.label is None
                        ), f"Prefix error:{code}->{current.right.label} & {y}"
                        self.replace(current.right, Node())
                    current = current.right

                # if c_i=0, we go down to the left child
                if code[i] == 0:
                    if isinstance(current.left, Leaf):
                        assert (
                            current.left.label is None
                        ), f"Prefix error:{code}->{current.right.label} & {y}"
                        self.replace(current.left, Node())
                    current = current.left

                # if c_i=-1, the current node should be the leaf
                if code[i] == -1:
                    break
            self.replace(current, self.y2leaf[y])

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

        pos_ind = y_codes[node.ind, node.depth] == 1

        # print(node)
        node.right.ind = node.ind.copy()
        node.right.ind[node.ind] = pos_ind
        node.right.value = np.sum(pos_ind)

        node.left.ind = node.ind.copy()
        node.left.ind[node.ind] = ~pos_ind
        node.left.value = node.value - node.right.value

    def find_admissible_partition(self, y_cat, epsilon=0, rng=np.random.default_rng()):
        """
        Number of queries need to find the empirical mode in a sequence

        Parameters
        ----------
        y_cat: list of int of size (n,)
            Sequential samples of the categorical variable
        epsilon: float, optional
            Stopping criterion for the difference in probability between the
            empirical mode and the sets in the found partition
        rng: numpy.random.Generator, optional
            Random number generator to break ties in comparision

        Returns
        -------
        partition: list
            List of nodes in the `epsilon`-admissible partition

        Notes
        -----
        This function sets tree nodes values to number of observation
        """
        y_codes = self.codes[y_cat]
        m = len(self.codes)
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
            # add noise to break ties arbitrarily in heap
            noise = rng.random(size=2) / ((2 * m)) ** 2
            # push children in the heap
            heapq.heappush(heap, (-node.left.value + noise[0], node.left))
            heapq.heappush(heap, (-node.right.value + noise[1], node.right))
        while len(heap) > 0:
            _, node = heapq.heappop(heap)
            partition.append(node)
        return partition

    def __repr__(self):
        return f"SearchTree at {id(self)}"
