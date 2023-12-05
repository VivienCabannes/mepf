import heapq

import numpy as np


# -----------------------------------------------------------------------------
# Graph Constructors
# -----------------------------------------------------------------------------


class Leaf:
    def __init__(self, value: int, label: int):
        self.value = value
        self.label = label
        self.parent = None
        self.depth = None

    def update_depth(self, depth):
        self.depth = depth

    def __repr__(self):
        return f"Leaf({self.value:3d}) at {id(self)}"

    def _get_print(self, _call=False):
        return [f"\033[1mLeaf {self.label}: {self.value:3d}\033[0m"]


class Node:
    def __init__(self, left: Leaf, right: Leaf):
        self.left = left
        self.right = right
        self.right.parent = self
        self.left.parent = self
        self.value = left.value + right.value
        self.parent = None
        self.depth = None

    def update_depth(self, depth):
        self.depth = depth
        self.left.update_depth(depth + 1)
        self.right.update_depth(depth + 1)

    def __repr__(self):
        return f"Node({self.value:3d}) at {id(self)}"

    def _get_print(self, _call=True):
        left_print = self.left._get_print(_call=False)
        right_print = self.right._get_print(_call=False)
        left_length, left_depth = len(left_print[0]), len(left_print)
        right_length, right_depth = len(right_print[0]), len(right_print)
        if isinstance(self.left, Leaf):
            left_length -= 8
        if isinstance(self.right, Leaf):
            right_length -= 8
        current = " " * (left_length - 3)
        current += f"Node: {self.value:3d}" + " " * (right_length - 3)
        if _call:
            out_print = current + "\n"
        else:
            out_print = [current]
        for i in range(max(left_depth, right_depth)):
            if i < left_depth:
                current = left_print[i]
            else:
                current = " " * left_length
            current += " | "
            if i < right_depth:
                current += right_print[i]
            else:
                current += " " * right_length
            if _call:
                out_print += current + "\n"
            else:
                out_print.append(current)
        return out_print


class Tree:
    def __init__(self, root: Node):
        self.root = root

    def __repr__(self):
        return f"HuffmanTree with root at {id(self.root)}"

    def __str__(self):
        return self.root._get_print()

    def update_depth(self):
        self.root.update_depth(0)


# -----------------------------------------------------------------------------
# Huffman tree
# -----------------------------------------------------------------------------


class HuffmanTree:
    """
    Huffman tree implementation

    Attributes
    ----------
    huffman_tree: Node
        The root of the Huffman tree.
    huffman_list: list of Leaf
        List of all nodes in the order they merged in the Huffman tree.
    y2leaf: dict of int: Leaf
        Dictionary mapping each class to its corresponding leaf.
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
        values += rng.random(size=m) / ((2 * m) ** 2)

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
        self.huffman_tree = Tree(node)

    def update(self, y):
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
        self._huffman_update(self.y2leaf[y])

    def _huffman_update(self, node, _i_min=0):
        """
        Report observation of the node, and update Huffman tree

        Parameters
        ----------
        node: Node
            Node to update
        huffman_list: list
            List of nodes sorted by value that used to build the Huffman tree
        _i_min: int
            Index of the last node in the list visited
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
        self._huffman_update(node.parent, i_swap)

    @staticmethod
    def swap(node1, node2):
        if node1.parent.left == node1:
            node1.parent.left = node2
        elif node1.parent.right == node1:
            node1.parent.right = node2
        if node2.parent.left == node2:
            node2.parent.left = node1
        elif node2.parent.right == node2:
            node2.parent.right = node1
        node1.parent, node2.parent = node2.parent, node1.parent
        node1.parent.update_depth(node1.parent.depth)
        node2.parent.update_depth(node2.parent.depth)

    def __repr__(self):
        return f"HuffmanTree at {id(self)}"

    def __str__(self):
        return self.huffman_tree.__str__()


# -----------------------------------------------------------------------------
# Huffman code implementation without graph objects
# -----------------------------------------------------------------------------


def _huffman_tree(frequencies):
    """
    Build a Huffman tree from the given frequencies.

    Parameters
    ----------
    frequencies : list of int
        Frequencies of the symbols.
    return_counts : bool
        If True, return the counts of each symbol.

    Returns
    -------
    children : dict
        Dictionary representing the children of each node.
    counts: dict
        Dictionary representing the counts of each symbol.
    """
    m = len(frequencies)
    heap = [(frequencies[i], i) for i in range(m)]
    heapq.heapify(heap)
    children = {}

    index = len(frequencies)
    while len(heap) > 1:
        min1, min2 = heapq.heappop(heap), heapq.heappop(heap)
        heapq.heappush(heap, (min1[0] + min2[0], index))
        children[index] = (min1[1], min2[1])
        index += 1
    return children


def _get_codes(children, prefix=[], root_index=None, codes={}):
    """
    Get the codes for each symbol from the Huffman tree recursively.

    Parameters
    ----------
    children : dict
        Dictionary representing the children of each node.
    prefix : list of int
        Prefix of the code.
    root_index : int
        Index of the root node.
    codes : dict
        Dictionary representing the codes of each symbol.

    Returns
    -------
    codes : dict
        Dictionary representing the codes of each symbol.
    """
    if root_index is None:
        # at initialization, we find root node as the one with maximal index
        root_index = max(children)
    if root_index not in children:
        # if the node is a leaf, we store its code which is the current prefix
        codes[root_index] = prefix
    else:
        # go down the left branch, and add 0 to the code prefix
        _get_codes(children, prefix + [0], children[root_index][0], codes)
        # go down the right branch, and add 1 to the code prefix
        _get_codes(children, prefix + [1], children[root_index][1], codes)
    return codes


def huffman_codes(frequencies):
    """
    Build a Huffman matrix from the given frequencies.

    Parameters
    ----------
    frequencies : list of int
        Frequencies of the symbols.

    Returns
    -------
    S : numpy.ndarray
        Huffman matrix. Each column represents the code of a symbol.
    """
    # build the Huffman tree
    children = _huffman_tree(frequencies)
    # explore the tree to get the codes for each elements
    codes = _get_codes(children)

    # write this code in matrix form
    M = max((len(codes[i]) for i in codes))
    m = len(frequencies)
    S = np.zeros((m, M), dtype=np.int8)
    S[:] = -1
    for i in range(m):
        S[i, : len(codes[i])] = codes[i]

    return S
