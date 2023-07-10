import heapq
import numpy as np


def build_Huffman_tree(frequencies):
    """
    Build a Huffman tree from the given frequencies.

    Parameters
    ----------
    frequencies : list of int
        Frequencies of the symbols.

    Returns
    -------
    children : dict
        Dictionary representing the children of each node.
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


def get_codes(children, prefix=[], root_index=None, codes={}):
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
        root_index = max(children)
    if root_index not in children:
        codes[root_index] = prefix
    else:
        get_codes(children, prefix + [0], children[root_index][0], codes)
        get_codes(children, prefix + [1], children[root_index][1], codes)


def Huffman_matrix(frequencies):
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
    children = build_Huffman_tree(frequencies)
    codes = {}
    get_codes(children, codes=codes)

    m = len(frequencies)
    M = max((len(codes[i]) for i in codes))
    S = np.zeros((M, m), dtype=np.int8)
    S[:] = -1
    for i in range(m):
        S[: len(codes[i]), i] = codes[i]

    return S
