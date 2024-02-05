import pytest

from entsearch.binary_tree import Node


class Helper:
    @staticmethod
    def reset_value(model):
        to_reset = [model.root]
        while len(to_reset) > 0:
            node = to_reset.pop(0)
            node.value = 0
            if type(node) is Node:
                to_reset.append(node.left)
                to_reset.append(node.right)


@pytest.fixture
def helpers():
    return Helper
