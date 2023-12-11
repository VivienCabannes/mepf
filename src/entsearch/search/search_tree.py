"""
Search Tree
"""
import heapq
from typing import List
import numpy as np
from .tree_constructors import Leaf, Node, Tree


class SearchTree(Tree):
    """
    Heuristic search implementation

    Attributes
    ----------
    partition: list of Vertex
        List of nodes in the current coarse partition.
    y2leaf: dict of int: Leaf
        Dictionary mapping each class to its corresponding leaf.
    y2node: dict of int: Node
        Dictionary mapping each class to its partition node.
    huffman_list: list of Leaf
        List of all nodes in the order they were merged in the Huffman tree.
    _i_part: int
        Index of the minimum partition node in the huffman list.
    mode: Vertex
        Current empirical mode at partition level.
    nb_queries: int
        Total number of queries made.
    y_cat: np.ndarray
        List of past samples.
    y_observations: np.ndarray
        List of weak observation of past samples.
    """

    def __init__(self, m: int, comeback: bool = False):
        """
        Initialize the tree.

        Parameters
        ----------
        m:
            Maximal number of potential class
        comeback:
            Wether to remember past information for future re-query
        batch:
            Wether we will find empirical mode in a batch
        """
        self.m = m

        # initialize the leaves and the partition
        self.partition = [Leaf(value=1, label=i) for i in range(m)]
        self.y2leaf = {i: self.partition[i] for i in range(m)}
        self.y2node = {}

        # build the Huffman tree
        root = self.huffman_build(self.partition)
        Tree.__init__(self, root)
        self.reset_value(0)

        # remember Huffman position
        self.huffman_list = self.get_huffman_list()
        for i, node in enumerate(self.huffman_list):
            node._i_huff = i
        self.partition_update()

        # initialize mode guess
        self.mode = self.y2node[0]
        self.nb_queries = 0

        if comeback:
            # remember past information for comeback
            self.y_cat = np.arange(m, dtype=int)
            self.y_observations = np.eye(m, dtype=bool)[np.arange(m)]

    def fine_identification(self, y: int, update: bool = True):
        """
        Get the precise label of y, and update the Huffman tree.

        Parameters
        ----------
        y:
            Class to update
        update:
            Wether to update the tree accordingly
        """
        node = self.y2leaf[y]
        self.nb_queries += node.depth
        if update:
            self._vitter_update(node)
        else:
            while node is not None:
                node.value += 1
                node = node.parent

    def coarse_identification(self, y: int, epsilon: float = 0):
        """
        Get the coarse information on y, and update the partition.

        Parameters
        ----------
        y:
            Class to update
        epsilon:
            Criterion on the biggest `p(S)` compared to `max p(y)`,
            for any non-singleton set :math:`p(S) < \\max_y p(y) - \\epsilon`
        """

        i = self.root.value
        # dynamic resizing of past history
        if i == len(self.y_cat):
            tmp = self.y_cat
            length = len(tmp)
            self.y_cat = np.zeros((2 * length), dtype=int)
            self.y_cat[:length] = tmp
            tmp = self.y_observations
            self.y_observations = np.zeros((2 * length, self.m), dtype=bool)
            self.y_observations[:length] = tmp

        # report observation
        node = self.y2node[y]
        self.nb_queries += node.depth
        self.y_cat[i] = y
        self.y_observations[i] = node._set_code

        # update the Huffman tree
        self._vitter_update(node)

        # see if we found the new empirical mode
        if self.mode < node:
            self.mode = node

        self._partition_rebalancing(epsilon)

    def batch_identification(
        self, y_cat: List[int], epsilon: float = 0, update: bool = True
    ):
        """
        Find the emprical mode in a batch

        Parameters
        ----------
        y_cat:
            Batch observation
        update:
            Wether to update the tree accordingly
        epsilon:
            Criterion on the biggest `p(S)` compared to `max p(y)`,
            for any non-singleton set :math:`p(S) < \\max_y p(y) - \\epsilon`
        """
        codes = self.get_codes()
        self.root.value = len(y_cat)
        setattr(self, "y_codes", codes[y_cat])
        self._refine_partition(epsilon)
        delattr(self, "y_codes")

        if update:
            root = self.huffman_build(self.partition)
            self.replace_root(root)

            # remember Huffman position
            self.huffman_list = self.get_huffman_list()
            for i, node in enumerate(self.huffman_list):
                node._i_huff = i

    def _partition_rebalancing(self, epsilon: float):
        """
        Rebalance partition

        Merge as much sets as possible while ensuring for non-singleton set `S`
        .. math::`p(S) < \\max_y p(y) - \\epsilon`

        Parameters
        ----------
        epsilon:
        """
        # if the partition mode is a not singleton, refine the partition
        if not hasattr(self.mode, "label"):
            self._splitting(epsilon)
            self._merging(epsilon)
        # or if there exists a coarser partition, find it
        else:
            min1 = self.huffman_list[self._i_part].value
            min2 = self.huffman_list[self._i_part + 1].value
            criterion = self.mode.value - epsilon * self.root.value
            if min1 + min2 < criterion:
                self._merging(epsilon)

    def _splitting(self, epsilon: float):
        codes = self.get_codes()
        setattr(self, "y_codes", codes[self.y_cat[:self.root.value]])
        self._refine_partition(epsilon)
        delattr(self, "y_codes")

    def _merging(self, epsilon: float):
        # we run Huffman at the partition level
        root = self.huffman_build(self.partition)
        self.replace_root(root)

        # remember Huffman position
        self.huffman_list = self.get_huffman_list()
        for i, node in enumerate(self.huffman_list):
            node._i_huff = i

        # update the partition, using the node value
        self._refine_partition(epsilon)

    def _refine_partition(self, epsilon: float):
        if hasattr(self, "y_codes"):
            # we are splitting nodes
            self.root.ind = np.ones(self.root.value, dtype=bool)
            n_mode = None
        else:
            # we are merging nodes
            n_mode = self.mode.value
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
            if hasattr(self, "y_codes"):
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

        # update the minimum partition node index
        self.partition_update()

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

        # TODO: report nb_queries

    def __repr__(self):
        return f"SearchTree at {id(self)}"

    def _vitter_update(self, node):
        """
        Update Huffman tree due to recent observation

        This method assimilates to dynamic Huffman coding (Vitter algorithm)

        Parameters
        ----------
        node: Node
            Node to update
        """
        # stop at the root node
        if node.parent is None:
            node.value += 1
            return

        # get the node and swap it with max equals elements
        i_node = node._i_huff
        i_swap = i_node
        while not node < self.huffman_list[i_swap + 1]:
            i_swap += 1
        if i_swap != i_node:
            self._swap(i_node, i_swap)
            i_node = i_swap

        # increase node value
        node.value += 1
        parent = node.parent

        # technicalities related to new nodes
        if parent.value == 0:
            if parent._i_huff != i_node + 1:
                # remove the node, and merge sibling and parent
                if parent.left == node:
                    self.replace(parent, parent.right)
                    parent = parent.right
                # be mindful to keep Huffman ordering
                else:
                    self._swap(parent.left._i_huff, i_node + 1)
                    self.replace(parent, parent.left)
                    parent = parent.left

                # get the highest node with no observation
                while parent.parent is not None and parent.parent.value == 0:
                    parent = parent.parent

                # set the node at this level
                grand_parent = parent.parent
                new = Node(parent, node)
                # if parent was the root, we have changed it
                if grand_parent is None:
                    self.replace_root(new)
                else:
                    # avoid recursion issue with depth computation
                    parent.parent = grand_parent
                    self.replace(parent, new)
                    parent.parent = new

                # rebuild the huffman list
                self.huffman_list = self.get_huffman_list()
                for i, node in enumerate(self.huffman_list):
                    node._i_huff = i

                # update the parent whose value has changed
                new.value -= 1
                parent = new
        else:
            # keep the huffman order by swapping the node again
            i_swap = i_node
            while self.huffman_list[i_swap + 1] < node:
                i_swap += 1
            if i_swap != i_node:
                swapped = self.huffman_list[i_swap]
                self._swap(i_node, i_swap)

                # update the parent whose value has changed
                if swapped.value != node.value:
                    parent = node.parent

        # update the parent node
        self._vitter_update(parent)

    def _swap(self, i_node, i_swap):
        node = self.huffman_list[i_node]
        swapped = self.huffman_list[i_swap]

        # update list and pointers
        self.huffman_list[i_node] = swapped
        swapped._i_huff = i_node
        self.huffman_list[i_swap] = node
        node._i_huff = i_swap

        # swap in the tree
        self.swap(node, swapped)

    def partition_update(self):
        """
        Update partition dictionary and set codes.
        """
        # update the partition dictionary
        self._i_part = np.inf
        for node in self.partition:
            y_set = self.get_leaves_set(node)
            for y in y_set:
                self.y2node[y] = node
            node._set_code = np.zeros(self.m, dtype=bool)
            node._set_code[y_set] = 1
            if self._i_part > node._i_huff:
                self._i_part = node._i_huff

    @staticmethod
    def get_leaves_set(node):
        """
        Get list of leaf descendants labels
        """
        if type(node) is Leaf:
            return [node.label]
        else:
            left_set = SearchTree.get_leaves_set(node.left)
            right_set = SearchTree.get_leaves_set(node.right)
            return left_set + right_set
