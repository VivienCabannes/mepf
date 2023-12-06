"""
Tree Constructors
"""
import heapq

import numpy as np


class Leaf:
    def __init__(self, value: int = 0, label: int = None):
        self.value = value
        self.label = label
        self.parent = None
        self.depth = None

    def update_depth(self, depth):
        self.depth = depth

    def reset_value(self):
        self.value = 0

    def get_set(self):
        self.set = [self.label]
        return self.set

    def delete_parent(self):
        parent = self.parent
        if parent is None:
            return
        parent.left = None
        parent.right = None
        parent.delete_parent()
        self.parent = None
        del parent

    def __repr__(self):
        return f"Leaf({self.value:3d}) at {id(self)}"

    def _get_print(self, _call=False, length=None):
        return [f"\033[1mLeaf {self.label}: {self.value:d}\033[0m"]


class Node:
    def __init__(self, left: Leaf = None, right: Leaf = None):
        if left is None:
            left = Leaf()
        if right is None:
            right = Leaf()

        self.left = left
        self.right = right
        self.left.parent = self
        self.right.parent = self
        self.value = self.left.value + self.right.value
        self.parent = None
        self.depth = None

    def update_depth(self, depth):
        self.depth = depth
        self.left.update_depth(depth + 1)
        self.right.update_depth(depth + 1)

    def reset_value(self):
        self.value = 0
        self.left.reset_value()
        self.right.reset_value()

    def get_set(self):
        self.set = self.left.get_set() + self.right.get_set()
        return self.set

    def delete_parent(self):
        parent = self.parent
        if parent is None:
            return
        parent.left = None
        parent.right = None
        parent.delete_parent()
        self.parent = None
        del parent

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
        # current = " " * (left_length - 3)
        # current += f"Node: {self.value:3d}" + " " * (right_length - 3)
        current = " " * (left_length - length)
        current += "Node: " + format(self.value, str(2 * length - 3) + "d")
        current += " " * (right_length - length)
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
        self.root.update_depth(0)

    def __repr__(self):
        return f"HuffmanTree with root at {id(self.root)}"

    def __str__(self):
        length = max(len(str(self.root.value)) // 2 + 1, 2)
        return self.root._get_print(length=length)

    def reset_value(self):
        self.root.reset_value()

    def replace_root(self, new_root):
        root = self.root
        self.root = new_root
        del root

    def init_from_codes(self, codes):
        """
        Initialize the tree from integer codes.

        Parameters
        ----------
        codes : list of list of int
            The codes of each class to initialize the tree.

        Returns
        -------
        y2leaf : dict of int: Leaf
            Dictionary mapping each class to its corresponding leaf.
        """
        Tree.__init__(self,  Node())
        y2leaf = {}

        # to build the search tree, iterate over the leaves
        for y in range(len(codes)):
            code = codes[y]
            y2leaf[y] = Leaf(0, y)
            current = self.root
            # iterate over the leaf code
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

                # if c_i=-1, we are at the position of the leaf
                if code[i] == -1:
                    break
            self.replace(current, y2leaf[y])
        return y2leaf

    def huffman_build(self, nodes, return_list=False, rng=np.random.default_rng()):
        """
        Build Huffman tree on top of nodes (seen as leaves)

        Parameters
        ----------
        node_list : list of Nodes
            The nodes to use as based leaves for the sup-tree.
        return_list : bool, optional
            Whether to return the Huffman list of nodes
        rng : numpy.random.Generator, optional
            The random number generator to use. The default is
            numpy.random.default_rng().

        Returns
        -------
        node: Node
            Root of the Huffman tree
        huffman_list: list, if `return_list`
            List of nodes in the order they were merged in the Huffman tree
        """
        # randomness to break ties in heap
        m = len(nodes)
        noise = rng.random(size=m) / ((2 * m)) ** 2

        # use heap to build the Huffman tree
        heap = [(nodes[i].value + noise[i], nodes[i]) for i in range(m)]
        heapq.heapify(heap)
        huffman_list = []
        while len(heap) > 1:
            left, right = heapq.heappop(heap), heapq.heappop(heap)
            huffman_list.append(left[1])
            huffman_list.append(right[1])
            node = Node(left[1], right[1])
            heapq.heappush(heap, (left[0] + right[0], node))
        if return_list:
            return node, huffman_list
        return node

    @staticmethod
    def swap(node1, node2):
        """
        Swapping two nodes in the tree

        Parameters
        ----------
        node1: Node
            First node to swap
        node2: Node
            Second node to swap
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
        depth1, depth2 = node1.depth, node2.depth
        node1.update_depth(depth2)
        node2.update_depth(depth1)

    @staticmethod
    def replace(node, new_node):
        """
        Replace a node by a new node

        Parameters
        ----------
        node: Node
            Node to replace
        new_node: Node
            New node to insert

        Notes
        -----
        Useful for SearchTree
        """
        if node.parent.left == node:
            node.parent.left = new_node
        elif node.parent.right == node:
            node.parent.right = new_node
        else:
            raise ValueError("Node not found in parent")
        new_node.parent = node.parent
        new_node.update_depth(node.depth)
        del node
