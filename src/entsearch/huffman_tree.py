"""
Huffman tree
"""
import numpy as np
from .tree_constructors import Tree, Leaf


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
        # add fake observations to get a balanced tree without observations
        m = len(counts)
        self.fake_addition = [counts[i] == 0 for i in range(m)]

        # initialize the leaves
        nodes = [
            Leaf(counts[i] + int(self.fake_addition[i]), label=i) for i in range(m)
        ]
        self.y2leaf = {i: nodes[i] for i in range(m)}

        # build the Huffman tree
        root, self.huffman_list = self.huffman_build(nodes, return_list=True, rng=rng)
        Tree.__init__(self, root)

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

        This method assimilates to dynamic Huffman coding (FGK algorithm)

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

    def get_nb_queries(self, y_cat):
        """
        Get the number of queries made to identify the empirical mode

        Parameters
        ----------
        y_cat : list of int
            Sequence of observations

        Returns
        -------
        nb_queries : int
            Number of queries make to identify the empirical mode
        """
        nb_queries = 0
        for y in y_cat:
            nb_queries += self.y2leaf[y].depth
            self.report_observation(y)
        return nb_queries

    def __repr__(self):
        return f"HuffmanTree at {id(self)}"
