"""
Tree Constructors
"""
import heapq

import numpy as np


class AllNode:
    def __init__(self):
        self.parent = None
        self.value = None
        pass

    def delete_parent(self):
        parent = self.parent
        if parent is None:
            return
        parent.left = None
        parent.right = None
        parent.delete_parent()
        self.parent = None
        del parent

    def __lt__(self, other):
        if self.value is None:
            return False
        if other.value is None:
            return True
        return self.value < other.value


class Leaf(AllNode):
    def __init__(self, value: int = None, label: int = None):
        self.value = value
        self.label = label
        self.parent = None
        self.depth = None

    def reset_value(self):
        self.value = None

    def update_depth(self, depth):
        self.depth = depth

    def get_max_depth(self):
        return self.depth

    def fill_codes(self, prefix, codes):
        codes[self.label] = prefix

    def __repr__(self):
        if self.value is None:
            return f"Leaf(None) at {id(self)}"
        return f"Leaf({self.value:3d}) at {id(self)}"

    def _get_print(self, _call=False, length=None):
        if self.value is None:
            return [f"\033[1mLeaf {self.label}: None\033[0m"]
        return [f"\033[1mLeaf {self.label}: {self.value:d}\033[0m"]


class Node(AllNode):
    def __init__(self, left: Leaf = None, right: Leaf = None):
        if left is None:
            assert right is None, "Cannot have only one child"
            self.left = Leaf()
            self.right = Leaf()
            self.value = None
        else:
            self.left = left
            self.right = right
            self.left.parent = self
            self.right.parent = self
            self.value = self.left.value + self.right.value
        self.parent = None
        self.depth = None

    def reset_value(self):
        self.value = None
        self.left.reset_value()
        self.right.reset_value()

    def update_depth(self, depth):
        self.depth = depth
        self.left.update_depth(depth + 1)
        self.right.update_depth(depth + 1)

    def get_max_depth(self):
        return max(self.left.get_max_depth(), self.right.get_max_depth())

    def get_nb_leaves(self):
        return self.left.get_nb_leaves() + self.right.get_nb_leaves()

    def fill_codes(self, prefix, codes):
        """
        Parameters
        ----------
        prefix : ndarray of size (max_depth,)
            The prefix of the code.
        codes : ndarray of size (m, max_depth)
            The list of codes to be filled.
        """
        right_prefix = prefix.copy()
        left_prefix = prefix.copy()
        right_prefix[self.depth] = 1
        left_prefix[self.depth] = 0
        self.right.fill_codes(right_prefix, codes)
        self.left.fill_codes(left_prefix, codes)

    def __repr__(self):
        return f"Node({self.value:3d}) at {id(self)}"

    def _get_print(self, _call=True, length=3):
        left_print = self.left._get_print(_call=False, length=length)
        right_print = self.right._get_print(_call=False, length=length)
        left_length, left_depth = len(left_print[0]), len(left_print)
        right_length, right_depth = len(right_print[0]), len(right_print)
        if isinstance(self.left, Leaf):
            left_length -= 8
        if isinstance(self.right, Leaf):
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


class Tree:
    def __init__(self, root: Node):
        self.root = root
        self.root.update_depth(0)

    def reset_value(self):
        self.root.reset_value()

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
            self.m = self.root.get_nb_leaves()
        length = self.get_depth()
        prefix = np.full(length, -1, dtype=int)
        codes = np.zeros((self.m, length), dtype=int)
        self.root.fill_codes(prefix, codes)
        return codes

    def __repr__(self):
        return f"HuffmanTree with root at {id(self.root)}"

    def __str__(self):
        out = ""
        for i in self.root._get_print(length=len(str(self.root.value))):
            out += i + "\n"
        return out

    def replace_root(self, new_root):
        root = self.root
        self.root = new_root
        self.root.update_depth(0)
        del root

    @staticmethod
    def swap(node1, node2, update_depth=False):
        """
        Swapping two nodes in the tree

        Parameters
        ----------
        node1: Node
            First node to swap
        node2: Node
            Second node to swap
        update_depth: bool, optional
            Whether to update the depth of the nodes after swapping

        Notes
        -----
        Useful for SearchTree
        """
        if node1.parent.left == node1:
            node1.parent.left = node2
        elif node1.parent.right == node1:
            node1.parent.right = node2
        if node2.parent.left == node2:
            node2.parent.left = node1
        elif node2.parent.right == node2:
            node2.parent.right = node1
        node1.parent, node2.parent = node2.parent, node1.parent

        if update_depth:
            depth1, depth2 = node1.depth, node2.depth
            node1.update_depth(depth2)
            node2.update_depth(depth1)

    @staticmethod
    def huffman_build(nodes, return_list=False):
        """
        Build Huffman tree on top of nodes (seen as leaves)

        Parameters
        ----------
        node_list : list of Nodes
            The nodes to use as based leaves for the sup-tree.
        return_list : bool, optional
            Whether to return the Huffman list of nodes

        Returns
        -------
        node: Node
            Root of the Huffman tree
        huffman_list: list, if `return_list`
            List of nodes in the order they were merged in the Huffman tree
        """
        # randomness to break ties in heap
        m = len(nodes)

        # use heap to build the Huffman tree
        heap = [nodes[i] for i in range(m)]
        heapq.heapify(heap)
        huffman_list = []
        while len(heap) > 1:
            left, right = heapq.heappop(heap), heapq.heappop(heap)
            huffman_list.append(left)
            huffman_list.append(right)
            node = Node(left, right)
            heapq.heappush(heap, node)
        if return_list:
            return node, huffman_list
        return node
