"""
Tree Constructors
"""


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
        current += "Node: " + format(self.value, str(2 * length - 3) + 'd')
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
        new_node.parent = node.parent
        new_node.update_depth(node.depth)
        del node
