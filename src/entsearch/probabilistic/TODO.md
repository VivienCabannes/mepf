

- try out the current elimination algorithm.
- implement the version with put-backs: add a counter for each node on the number of total knowledge. Call it time, and change the variable name on overleaf accordingly
- implement the set elimination versions
    - per batch
    - one sample at the time
    - with soft elimination
- implement a version where the eliminated nodes form a set put in the Huffman tree.

Notes
-----

Elimination algorithm.

Implementation choice:
- we query leaves.
- we query nodes.

choice:
- we eliminate forever.
- we put back in the game.

choice:
- we have an elimination set at the top.
- we have an elimination set in the huffman tree itself.

choice:
- only one sample at the time.
- per batch of data.

Implementation ordering:
1.1.1. - from exhaustive search
2.1.1. - from truncated search
1.2.1. - from truncated search with comeback

Huffman tree: correspond to a different y2node function in truncated search.