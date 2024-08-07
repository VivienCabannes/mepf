"""
Tree Constructors
"""

from __future__ import annotations

import heapq
from typing import List

import numpy as np


class Vertex:
    """
    Vertex object

    Attributes
    ----------
    value: int
        Empirical counts
    code: ndarray of size (max_depth,)
        Code associated with the vertex
    """

    def __init__(self):
        self.parent: Vertex = None
        self.depth: int = None
        self.value: int = None

    def get_descendent_labels(self):
        pass

    def get_set_code(self, m):
        set_code = np.zeros(m, dtype=bool)
        y_set = self.get_descendent_labels()
        set_code[y_set] = 1
        return set_code

    def __lt__(self, other: Vertex):
        """
        Vitter order.
        """
        # None value is seen a +inf
        if self.value is None:
            return False
        if other.value is None:
            return True
        if self.value < other.value:
            return True
        if self.value > other.value:
            return False
        if type(self) is type(other):
            return False
        if isinstance(self, Node) and not self.is_leaf_node:
            return False
        if isinstance(self, Node) and self.is_leaf_node and isinstance(other, Leaf):
            return False
        return True


class Leaf(Vertex):
    """
    Leaf object

    Attributes
    ----------
    label: int
        Class associated with this leaf
    """

    def __init__(self, value: int = None, label: int = None):
        Vertex.__init__(self)
        self.value = value
        self.label = label
        self.is_leaf_node = None

    def update_depth(self, depth: int):
        self.depth = depth

    def get_max_depth(self):
        return self.depth

    def get_descendent_labels(self):
        return [self.label]

    def fill_codes(self, prefix: np.ndarray, codes: np.ndarray):
        self.code = prefix
        codes[self.label] = prefix

    def __repr__(self):
        if self.value is None:
            return f"Leaf(None) at {id(self)}"
        return f"Leaf({self.value:3d}) at {id(self)}"

    def _get_print(self, length=None):
        if self.value is None:
            return [f"\033[1mLeaf {self.label}: None\033[0m"]
        return [f"\033[1mLeaf {self.label}: {self.value:d}\033[0m"]


class Node(Vertex):
    """
    Node object
    """

    def __init__(self, left: Vertex = None, right: Vertex = None):
        Vertex.__init__(self)
        if left is None:
            assert right is None, "Cannot have only one child"
            self.left = Leaf()
            self.right = Leaf()
        else:
            self.left = left
            self.right = right
            self.left.parent = self
            self.right.parent = self

        self.is_leaf_node = False

        if self.left.value is not None and self.right.value is not None:
            self.value = self.left.value + self.right.value

    def update_depth(self, depth: int):
        self.depth = depth
        self.left.update_depth(depth + 1)
        self.right.update_depth(depth + 1)

    def get_max_depth(self):
        return max(self.left.get_max_depth(), self.right.get_max_depth())

    def get_descendent_labels(self):
        return self.left.get_descendent_labels() + self.right.get_descendent_labels()

    def fill_codes(self, prefix: np.ndarray, codes: np.ndarray):
        """
        Parameters
        ----------
        prefix : ndarray of size (max_depth,)
            The prefix of the code.
        codes : ndarray of size (m, max_depth)
            The list of codes to be filled.
        """
        self.code = prefix
        right_prefix = prefix.copy()
        left_prefix = prefix.copy()
        right_prefix[self.depth] = 1
        left_prefix[self.depth] = 0
        self.right.fill_codes(right_prefix, codes)

        self.left.fill_codes(left_prefix, codes)

    def __repr__(self):
        if self.value is None:
            return f"Node(None) at {id(self)}"
        return f"Node({self.value:3d}) at {id(self)}"

    def __str__(self):
        out = ""
        for i in self._get_print(length=len(str(self.value))):
            out += i + "\n"
        return out

    def _get_print(self, length: int = 3):
        left_print = self.left._get_print(length=length)
        right_print = self.right._get_print(length=length)
        left_length, left_depth = len(left_print[0]), len(left_print)
        right_length, right_depth = len(right_print[0]), len(right_print)
        if isinstance(self.left, Leaf) or isinstance(self.left, EliminatedNode):
            left_length -= 8
        if isinstance(self.right, Leaf) or isinstance(self.right, EliminatedNode):
            right_length -= 8
        if self.value is None:
            current = " " * (left_length - 4)
            current += "Node: None"
            current += " " * (right_length - 3)
        else:
            current = " " * (left_length - 2 - length // 2)
            current += "Node: " + format(self.value, str(length) + "d")
            current += " " * (right_length - 1 - length // 2 - length % 2)
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
            out_print.append(current)
        return out_print


class EliminatedNode(Vertex):
    def __init__(self, eliminated_leaves: List[Vertex] = None):
        Vertex.__init__(self)
        if eliminated_leaves is None:
            self.children = []
        else:
            self.children = eliminated_leaves
        self.value = 0
        for child in self.children:
            self.value += child.value

    def update_depth(self, depth: int):
        self.depth = depth

    def get_max_depth(self):
        return self.depth

    def get_descendent_labels(self):
        labels = []
        for child in self.children:
            labels += child.get_descendent_labels()
        return labels

    def fill_codes(self, prefix: np.ndarray, codes: np.ndarray):
        self.code = prefix
        labels = self.get_descendent_labels()
        codes[labels] = prefix

    def __repr__(self):
        if self.value is None:
            return f"EliminatedNode(None) at {id(self)}"
        return f"EliminatedNode({self.value:3d}) at {id(self)}"

    def __str__(self):
        out = f"EliminatedNode: {self.value} \n"
        strs = [child._get_print() for child in self.children]
        lengths = [len(i[0]) for i in strs]
        lines_nb = max(len(i) for i in strs)
        for i in range(lines_nb):
            for j, tmp in enumerate(strs):
                if tmp:
                    cur = tmp.pop(0)
                else:
                    cur = " " * lengths[j]
                out += cur + " | "
            out = out[:-3]
            out += "\n"
        return out

    def _get_print(self, length=None):
        label = self.get_descendent_labels()
        if self.value is None:
            return [f"\033[1mElim {label}: None\033[0m"]
        return [f"\033[1mElim {label}: {self.value:d}\033[0m"]


class Tree:
    def __init__(self, root: Vertex):
        self.root = root
        self.root.update_depth(0)

    def get_depth(self):
        return self.root.get_max_depth()

    def get_codes(self):
        """
        Get the codes of the leaves associated with the tree

        Returns
        -------
        codes : ndarray of size (m, max_depth)
            The list of codes
        """
        if not hasattr(self, "m"):
            self.m = len(self.root.get_descendent_labels())
        length = self.get_depth()
        prefix = np.full(length, -1, dtype=int)
        codes = np.full((self.m, length), -1, dtype=int)
        self.root.fill_codes(prefix, codes)
        return codes

    def __repr__(self):
        return f"HuffmanTree with root at {id(self.root)}"

    def __str__(self):
        return self.root.__str__()

    def replace_root(self, new_root: Vertex):
        root = self.root
        self.root = new_root
        self.root.update_depth(0)
        del root

    @staticmethod
    def swap(node1: Vertex, node2: Vertex):
        """
        Swapping two nodes in the tree

        Parameters
        ----------
        node1: Node
            First node to swap
        node2: Node
            Second node to swap

        Notes
        -----
        Useful for HuffmanTree
        """
        if node1.parent == node2.parent:
            parent = node1.parent
            right, left = parent.right, parent.left
            parent.left = right
            parent.right = left
        else:
            if node1.parent.left == node1:
                node1.parent.left = node2
            elif node1.parent.right == node1:
                node1.parent.right = node2
            if node2.parent.left == node2:
                node2.parent.left = node1
            elif node2.parent.right == node2:
                node2.parent.right = node1
            node1.parent, node2.parent = node2.parent, node1.parent

            # update depth
            depth1, depth2 = node1.depth, node2.depth
            node1.update_depth(depth2)
            node2.update_depth(depth1)

    @staticmethod
    def huffman_build(nodes: List[Vertex]):
        """
        Build Huffman tree on top of nodes (seen as leaves)

        Parameters
        ----------
        node_list : list of Nodes
            The nodes to use as based leaves for the sup-tree.

        Returns
        -------
        node: Node
            Root of the Huffman tree
        """
        heap = nodes.copy()
        heapq.heapify(heap)
        while len(heap) > 1:
            left, right = heapq.heappop(heap), heapq.heappop(heap)
            heapq.heappush(heap, Node(left, right))
        return heap[0]

    @staticmethod
    def build_balanced_subtree(nodes: List[Vertex]):
        """
        Build balanced subtree with specified vertex.
        """
        for node in nodes:
            node.value = 1
        root = Tree.huffman_build(nodes)
        node_list = [root]
        while len(node_list) > 0:
            node = node_list.pop(0)
            node.value = 0
            if type(node) is Node:
                node_list.append(node.left)
                node_list.append(node.right)
        return root

    def get_huffman_list(self, partition=False):
        """
        Get the list of nodes in the order used to build the Huffman tree

        Returns
        -------
        huffman_list : list of Nodes
            The list of nodes in the order used to build the Huffman tree
        partition: bool
            Whether to return the Huffman list at partition level

        Notes
        -----
        We do not do it during the `huffman_build` method due to inconsistent ties breaking in the heap.
        This method gives a sorted list where leaves are understood at the partition level.
        """
        codes = self.get_codes()

        if partition:
            for node in self.partition:
                labels = node.get_descendent_labels()
                codes[labels] = -1
                codes[labels[0]] = node.code

        # order codes by depth
        depth = codes.shape[1] + 1
        codes_per_depth = {i: set({}) for i in range(depth + 1)}
        for code in codes:
            current = ""
            for char in code:
                if char == -1:
                    break
                current += str(char)
                codes_per_depth[len(current)].add(current)

        # sort codes from left to right
        sorted_nodes = []
        for i in range(depth + 1):
            sorted_nodes = sorted(codes_per_depth[i]) + sorted_nodes

        # deduce huffman list
        huffman_list = []
        for code in sorted_nodes:
            node = self.root
            for char in code:
                if char == "0":
                    node = node.left
                else:
                    node = node.right
            huffman_list.append(node)
        huffman_list.append(self.root)

        return huffman_list
