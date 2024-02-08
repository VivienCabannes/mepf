"""
Exhaustive Tree Search
"""
from ..binary_tree import Leaf, Node, Tree


class ExhaustiveSearch(Tree):
    """
    Exhaustive Tree Search

    Attributes
    ----------
    y2leaf: dict of int: Leaf
        Dictionary mapping each class to its corresponding leaf.
    huffman_list: list of Leaf
        List of all nodes in the order they were merged in the Huffman tree.
    nb_queries: int
        Total number of queries made.
    """

    def __init__(self, m: int, adaptive: bool = False):
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

        # initialize leaves mappings
        self.y2leaf = {i: Leaf(value=0, label=i) for i in range(m)}

        # initialize the tree
        root = Tree.build_balanced_subtree(list(self.y2leaf.values()))
        Tree.__init__(self, root)

        # remember Huffman position
        self.update_huffman_list()

        # initialize mode guess
        self.nb_queries = 0
        self.mode = self.y2leaf[0]

    def __call__(self, y: int):
        """
        Get the precise label of y, and update the Huffman tree.

        Parameters
        ----------
        y:
            Class to update
        """
        node = self.y2leaf[y]
        self.nb_queries += node.depth
        if self.adaptive:
            self._vitter_update(node)
        else:
            while node is not None:
                node.value += 1
                node = node.parent
            node = self.y2leaf[y]

        if self.mode < node:
            self.mode = node

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

        # get the node and swap it with max equal elements
        i_node = node._i_huff
        i_swap = i_node
        while not node < self.huffman_list[i_swap + 1]:
            i_swap += 1
        if i_swap != i_node:
            self._swap(i_node, i_swap)
            i_node = i_swap

        # technicalities related to new nodes
        if type(node) is Leaf and node.value == 0:
            self._nyo_update(node)
            return

        # increase node value
        node.value += 1
        parent = node.parent

        # keep the huffman order by swapping the node again
        node.parent.value += 1
        i_swap = i_node
        while self.huffman_list[i_swap + 1] < node:
            i_swap += 1
        node.parent.value -= 1
        if i_swap != i_node:
            swapped = self.huffman_list[i_swap]
            self._swap(i_node, i_swap)

            # update the parent whose value has changed
            if swapped.value != node.value:
                parent = node.parent

        # update the parent node
        self._vitter_update(parent)

    def _nyo_update(self, leaf):
        """
        Dealing with new observations
        """
        # find parent with no observation
        node = leaf
        while node.parent is not None and node.parent.value == 0:
            node = node.parent

        leaves_set = set(node.get_descendent_labels())
        leaves_set.remove(leaf.label)
        if len(leaves_set) > 1:
            grand_parent = node.parent
            # create new `not yet observed`` balanced tree
            nyo_leaves = [self.y2leaf[i] for i in leaves_set]
            nyo = Tree.build_balanced_subtree(nyo_leaves)
            # slide newly observed leaf up in the tree
            new_node = Node(nyo, leaf)
            if grand_parent is None:
                self.replace_root(new_node)
            else:
                new_node.parent = grand_parent
                if grand_parent.left == node:
                    grand_parent.left = new_node
                elif grand_parent.right == node:
                    grand_parent.right = new_node
                new_node.update_depth(node.depth)
                del node

            # update attributes
            self.update_huffman_list()

        leaf.value += 1
        self._vitter_update(leaf.parent)

    def _swap(self, i_node, i_swap):
        """
        Swap two nodes in the Huffman tree.
        """
        node = self.huffman_list[i_node]
        swapped = self.huffman_list[i_swap]

        # update list and pointers
        self.huffman_list[i_node] = swapped
        swapped._i_huff = i_node
        self.huffman_list[i_swap] = node
        node._i_huff = i_swap

        # swap in the tree
        self.swap(node, swapped)

    def update_huffman_list(self):
        """
        Update model and nodes attributes to remember Huffman positions.
        """
        self.huffman_list = self.get_huffman_list()
        for i, node in enumerate(self.huffman_list):
            node._i_huff = i

    def __repr__(self):
        return f"ExhaustiveSearchTree at {id(self)}"
