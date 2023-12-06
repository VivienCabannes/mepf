"""
Huffman code (implemented without graph objects)
"""
import heapq

import numpy as np


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
