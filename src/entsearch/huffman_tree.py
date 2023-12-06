"""
Huffman tree
"""
import heapq
import numpy as np
from .tree_constructors import Tree, Node, Leaf


class HuffmanTree(Tree):
    """
    Huffman tree implementation

    Attributes
    ----------
    root: Node
        The root of the Huffman tree.
    y2leaf: dict of int: Leaf
        Dictionary mapping each class to its corresponding leaf.
    huffman_list: list of Leaf
        List of all nodes in the order they were merged in the Huffman tree.
    """

    def __init__(self, counts, rng=np.random.default_rng()):
        """
        Initialize the Huffman tree with integer counts.

        Parameters
        ----------
        counts : list of int
            The counts of each class to initialize the tree.
        rng : numpy.random.Generator, optional
            The random number generator to use. The default is
            numpy.random.default_rng().
        """
        # preprocess the counts before the Huffman tree construction
        m = len(counts)
        values = np.array([float(counts[i]) for i in range(m)])
        #    add fake observations to get a balanced tree without observations
        self.fake_addition = values == 0
        values[self.fake_addition] += 1
        #    add randomness to break ties in heap
        values += rng.random(size=m) / ((2 * m)) ** 2

        # initialize the leaves
        self.y2leaf = {i: Leaf(int(values[i]), label=i) for i in range(m)}

        # use heap to build the Huffman tree
        heap = [(values[i], self.y2leaf[i]) for i in range(m)]
        heapq.heapify(heap)
        self.huffman_list = []
        while len(heap) > 1:
            left, right = heapq.heappop(heap), heapq.heappop(heap)
            self.huffman_list.append(left[1])
            self.huffman_list.append(right[1])
            node = Node(left[1], right[1])
            heapq.heappush(heap, (left[0] + right[0], node))
        Tree.__init__(self, node)

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
        self._update(self.y2leaf[y])

    def _update(self, node, _i_min=0):
        """
        Report observation of the node, and update Huffman tree

        Parameters
        ----------
        node: Node
            Node to update
        _i_min: int
            Index of the last node in the list visited

        Notes
        -----
        The update algorithm makes use of the huffman_list,
        which sorts nodes by value, in the order used to build the Huffman tree
        """
        # if we are at the root node, we stop
        if node.parent is None:
            node.value += 1
            return

        # get the node
        i_node = _i_min
        while i_node < len(self.huffman_list):
            if self.huffman_list[i_node] == node:
                break
            i_node += 1
        assert i_node < len(
            self.huffman_list
        ), f"{node} not found in list after {_i_min}."

        # find where to swap it in the list
        i_swap = i_node + 1
        while i_swap < len(self.huffman_list):
            if self.huffman_list[i_swap].value != node.value:
                break
            i_swap += 1
        # deduce the node to swap with
        i_swap -= 1

        if i_swap != i_node:
            swapped = self.huffman_list[i_swap]
            # swap them in the tree and in the sorted list
            self.huffman_list[i_node] = swapped
            self.huffman_list[i_swap] = node
            self.swap(node, swapped)

        # update the node value and iter over its parents
        node.value += 1
        self._update(node.parent, i_swap)

    def __repr__(self):
        return f"HuffmanTree at {id(self)}"
