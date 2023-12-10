"""
Search Tree
"""
import heapq
import numpy as np
from .huffman_tree import HuffmanTree
from .tree_constructors import Leaf


class HeuristicSearch(HuffmanTree):
    """
    Heuristic search implementation

    Attributes
    ----------
    y2partition: dict of int: Node
        Dictionary mapping each class to its partition node.
    partition: list of Vertex
        List of nodes in the current partition.
    i_part: int
        Index of the minimum partition node in the huffman list.
    mode: Vertex
        Current empirical mode at partition level.
    nb_queries: int
        Number of queries made to identify the empirical mode.

    See Also
    --------
    HuffmanTree and HuffmanTree Attributes.
    """

    def __init__(self, counts):
        """
        Initialize the tree.

        Parameters
        ----------
        counts : list of int
            The counts of each class to initialize the tree.
        """
        HuffmanTree.__init__(self, counts)

        # we start with the trivial partition
        self.y2partition = {}
        self.partition = list(self.y2leaf.values())
        self.partition_update()
        self.i_part = 0

        self.mode = self.y2partition[0]
        self.nb_queries = 0

        self.y_cat = np.arange(self.m, dtype=int)
        self.y_observations = np.eye(self.m, dtype=np.bool)[np.arange(self.m)]

    def report_observation(self, y):
        """
        Report observation of the class y, and update the Huffman tree

        Parameters
        ----------
        y: int
            Class to update
        """
        # If we had count a fake observation before, just make it a real one
        if self.fake_addition[y]:
            self.fake_addition[y] = False
            return
        i = self.root.value
        # dynamic resizing of past history
        if i == len(self.y_cat):
            tmp = self.y_cat
            length = len(tmp)
            self.y_cat = np.zeros((2 * length), dtype=int)
            self.y_cat[:length] = tmp
            tmp = self.y_observations
            self.y_observations = np.zeros((2 * length, self.m), dtype=np.bool)
            self.y_observations[:length] = tmp
        self.y_cat[i] = y
        self.y_observations[i] = self.y2partition[y].set_code
        self._coarse_update(y)
        self._partition_rebalancing()

    def _coarse_update(self, y):
        """
        Report observation of the class y, and update the Huffman tree

        Parameters
        ----------
        y: int
            Sample observed
        """
        # Find the node in the partition
        node = self.y2partition[y]
        self.nb_queries += node.depth

        # update the Huffman tree
        self._update(node)

        # see if we found the new empirical mode
        if self.mode.value < node.value:
            self.mode = node

    def _partition_rebalancing(self):
        # if the partition mode is a not singleton, refine the partition
        if not hasattr(self.mode, "label"):
            self._splitting()

        # if there exists a coarser partition, find it
        min1 = self.huffman_list[self.i_part].value
        min2 = self.huffman_list[self.i_part + 1].value
        if min1 + min2 < self.mode.value:
            self._merging()

    def _splitting(self):
        codes = self.get_codes()
        setattr(self, 'y_codes', codes[self.y_cat[:self.root.value]])
        self._refine_partition()
        delattr(self, 'y_codes')

    @staticmethod
    def _report_count(node, y_codes):
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

    def _merging(self):
        # we run Huffman at the partition level
        root = self.huffman_build(self.partition)
        self.replace_root(root)

        # remember Huffman position
        self.huffman_list = self.get_huffman_list()
        for i, node in enumerate(self.huffman_list):
            node._i_huff = i

        # we report the count for the newly merged nodes
        for node in self.partition:
            setattr(node, "partition_mark", True)
        self.get_partition_values(self.root)
        for node in self.partition:
            delattr(node, "partition_mark")

        # update the partition, using the node value
        self._refine_partition()

    def _refine_partition(self):
        if hasattr(self, 'y_codes'):
            self.root.ind = np.ones(self.root.value, dtype=bool)
            n_mode = None
        else:
            n_mode = self.mode.value

        # update the partition
        heap = []
        heapq.heappush(heap, (-self.root.value, self.root))
        self.partition = []
        while len(heap) > 0:
            # get the node with the biggest value
            _, node = heapq.heappop(heap)
            # if are at the mode, we stop
            if n_mode is not None and node.value <= n_mode:
                self.partition.append(node)
                break
            # the first leaf we find if the mode
            if isinstance(node, Leaf):
                self.partition.append(node)
                assert n_mode is None
                self.mode = node
                n_mode = node.value
                break
            if hasattr(self, 'y_codes'):
                # make n_node queries to get children information
                self._report_count(node, self.y_codes)
            # push children in the heap
            heapq.heappush(heap, (-node.left.value, node.left))
            heapq.heappush(heap, (-node.right.value, node.right))
        # the remaning node form the partition
        while len(heap) > 0:
            _, node = heapq.heappop(heap)
            self.partition.append(node)

        # update the minimum partition node index
        self.i_part = node._i_huff
        self.partition_update()

    @staticmethod
    def get_partition_values(node):
        if hasattr(node, "partition_mark"):
            return node.value
        else:
            left_value = HeuristicSearch.get_partition_values(node.left)
            right_value = HeuristicSearch.get_partition_values(node.right)
            node.value = left_value + right_value
            return node.value

    def partition_update(self):
        # update the partition dictionary
        for node in self.partition:
            y_set = self.get_leaves_set(node)
            for y in y_set:
                self.y2partition[y] = node
            node.set_code = np.zeros(self.m, dtype=bool)
            node.set_code[y_set] = 1

    @staticmethod
    def get_leaves_set(node):
        """
        Get list of leaf descendants labels
        """
        if type(node) is Leaf:
            return [node.label]
        else:
            left_set = HeuristicSearch.get_leaves_set(node.left)
            right_set = HeuristicSearch.get_leaves_set(node.right)
            return left_set + right_set

    def __repr__(self):
        return f"SearchTree at {id(self)}"
