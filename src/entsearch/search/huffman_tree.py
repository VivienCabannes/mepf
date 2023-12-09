"""
Huffman tree
"""
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

    def __init__(self, counts):
        """
        Initialize the Huffman tree.

        Parameters
        ----------
        counts : list of int or float
            Counts of each class to initialize the tree.
        """
        # add fake observations to get a balanced tree without observations
        self.m = len(counts)
        self.fake_addition = [counts[i] == 0 for i in range(self.m)]

        # initialize the leaves
        nodes = []
        for i in range(self.m):
            leaf = Leaf(value=counts[i] + int(self.fake_addition[i]), label=i)
            nodes.append(leaf)
        self.y2leaf = {i: nodes[i] for i in range(self.m)}

        # build the Huffman tree
        root = self.huffman_build(nodes)
        Tree.__init__(self, root)

        # remember Huffman position
        self.huffman_list = self.get_huffman_list()
        for i, node in enumerate(self.huffman_list):
            node._i_huff = i

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

    def _update(self, node):
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
        value = node.value
        node.value += 1

        # keep the huffman order by swapping the node again
        i_swap = i_node
        while self.huffman_list[i_swap + 1] < node:
            i_swap += 1
        if i_swap != i_node:
            self._swap(i_node, i_swap)

        # update the parent of the node whose value has changed
        if self.huffman_list[i_node].value == value:
            self._update(node.parent)
        else:
            self._update(self.huffman_list[i_node].parent)

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
