from typing import Union


class Leaf:
    """
    Leaf object

    Parameters
    ----------
    element: int
        The element (e.g. class) of the leaf
    proba: float
        The probability of the class.
    """

    def __init__(self, element: int, proba: float):
        self.element = element
        self.proba = proba

    def __repr__(self) -> str:
        """
        Print the leaf.
        """
        return "Leaf " + str(self.element) + " (" + str(self.proba) + ")"

    def set_code(self, prefix, code_dict=None):
        self.code = prefix
        if code_dict is not None:
            code_dict[self.element] = self.code


class Node:
    """
    Node object

    Parameters
    ----------
    child1: Leaf or Node
        The left child of the node.
    child2: Leaf or Node
        The right child of the node.
    """

    def __init__(self, child1, child2):
        self.right = child1
        self.left = child2
        self.proba = child1.proba + child2.proba

    def __repr__(self) -> str:
        """
        Print the node.
        """
        return "Node (" + str(self.proba) + ")"

    def set_code(self, prefix, code_dict=None):
        self.right.set_code(prefix + "0", code_dict=code_dict)
        self.left.set_code(prefix + "1", code_dict=code_dict)


class ListElement:
    """
    Element object of chained list

    Parameters
    ----------
    data: Leaf or Node
        The data of the element.
    """

    def __init__(self, data: Union[Leaf, Node]):
        self.data = data
        self.next_element = None

    def __repr__(self) -> str:
        return self.data.__repr__()


class ChainList:
    """
    Chained list object

    Parameters
    ----------
    root_element: ListElement
        The root element of the chained list.
    """

    def __init__(self, root_element: ListElement = None):
        self.next_element = root_element

    def add_element(self, element: ListElement):
        """
        Add an element to the chained list.

        Parameters
        ----------
        element: ListElement
            The element to add to the chained list.
        """
        old = self
        current = old.next_element
        while current is not None:
            if current.data.proba >= element.data.proba:
                element.next_element = current
                old.next_element = element
                return
            old = current
            current = current.next_element
        old.next_element = element

    def pop(self):
        """
        Pop root element from the list
        """
        element = self.next_element
        self.next_element = element.next_element
        return element

    def __repr__(self) -> str:
        """
        Print the chained list.
        """
        current = self.next_element
        string = ""
        if current is None:
            return string
        while current is not None:
            string += current.__repr__() + " -> "
            current = current.next_element
        return string[:-4]


def huffman_algorithm(proba):
    """
    Algorithm to build a Huffman code.

    Parameters
    ----------
    proba: list of float
        The list of probabilities of each leaf.

    Returns
    -------
    code_dict: dict
        The dictionary of the code of each leaf.
    """
    # Init queue with insertion sort on leaf probabilities
    queue = ChainList()
    for i in range(len(proba)):
        node = Leaf(i, proba=proba[i])
        queue.add_element(ListElement(node))

    # While there is more than one node in the queue:
    while queue.next_element.next_element is not None:
        # Remove the two nodes with lowest probability from the queue.
        node1 = queue.pop().data
        node2 = queue.pop().data
        # Merge those two nodes into a bigger one to insert into the queue.
        queue.add_element(ListElement(Node(node1, node2)))

    # Once the tree is built, enumerate the tree to find the code of each leaf.
    code_dict = {}
    queue.next_element.data.set_code("", code_dict=code_dict)
    return code_dict
