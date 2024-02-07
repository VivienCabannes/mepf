"""
Search Tree to solve the Mode Estimation with Partial Feedback problem.
"""
import heapq
import numpy as np
from ..binary_tree import EliminatedNode, Leaf, Node, Tree


class SetElimination(Tree):
    """
    Forever Set Elimination

    Attributes
    ----------
    partition: list of Vertex
        List of nodes in the current coarse partition (excluding the trash set).
    y2leaf: dict of int: Leaf
        Dictionary mapping each class to its corresponding leaf.
    y2node: dict of int: Node
        Dictionary mapping each class to its partition node.
    huffman_list: list of Leaf
        List of all nodes in the order they were merged in the Huffman tree.
    _i_part: int
        Index of the minimum partition node in the huffman list.
    trash: EliminatedNode
        Node to store eliminated nodes.
    mode: Vertex
        Current empirical mode at partition level.
    nb_queries: int
        Total number of queries made.
    y_cat: np.ndarray
        List of past samples.
    y_observations: np.ndarray
        List of weak observation of past samples.
    """

    def __init__(self, m: int, confidence_level: float = 1, constant: float = 24):
        """
        Initialize the tree.

        Parameters
        ----------
        m:
            Maximal number of potential class
        confidence_level:
            Level of certitude require to eliminate guesses
        constant:
            Constant to resize confidence intervals, default is 24.
        """
        self.m = m

        # elimination criterion
        self.delta = 1 - confidence_level
        self.constant = constant

        # initialize leaves mappings
        self.y2leaf = {i: Leaf(value=0, label=i) for i in range(m)}
        self.y2node = {}
        self.trash = EliminatedNode()
        self.eliminated = np.zeros(m, dtype=bool)

        # initialize the tree
        root = Tree.build_balanced_subtree(list(self.y2leaf.values()))
        Tree.__init__(self, root)

        # initialize the partition
        self.partition = [root]

        # remember Huffman position
        self.huffman_list = self.get_huffman_list()
        for i, node in enumerate(self.huffman_list):
            node._i_huff = i
            node.value = 0
        self.partition_update()

        # initialize mode guess
        self.mode = self.y2node[0]
        self.nb_queries = 0

        # remember past information for comeback
        self.y_cat = np.arange(m, dtype=int)
        self.y_observations = np.eye(m, dtype=bool)[np.arange(m)]

    def __call__(self, y: int, epsilon: float = 0):
        """
        Get the coarse information on y, and update the partition.

        Parameters
        ----------
        y:
            Class to update
        epsilon:
            Criterion on the biggest `p(S)` compared to `max p(y)`,
            for any non-singleton set :math:`p(S) < \\max_y p(y) - \\epsilon`

        See Also
        --------
        TruncatedSearch
        """
        # if we have already eliminated all nodes, we do nothing
        if self.eliminated.sum() == self.m - 1:
            return

        i = self.root.value

        # dynamic resizing of past history
        if i == len(self.y_cat):
            tmp = self.y_cat
            length = len(tmp)
            self.y_cat = np.zeros((2 * length), dtype=int)
            self.y_cat[:length] = tmp
            tmp = self.y_observations
            self.y_observations = np.zeros((2 * length, self.m), dtype=bool)
            self.y_observations[:length] = tmp

        # report observation
        node = self.y2node[y]
        self.nb_queries += node.depth
        self.y_cat[i] = y
        self.y_observations[i] = node._set_code

        # if y is already eliminated, we stop here
        if isinstance(node, EliminatedNode):
            node.value += 1
            return

        # update the Huffman tree
        self._vitter_update(node)

        # see if we found the new empirical mode
        if self.mode < node:
            self.mode = node

        self._partition_rebalancing(epsilon)

        # check for elimination at partition level
        if self.delta:
            sigma = np.log(((np.pi * self.root.value) ** 2) * self.m / self.delta)
            sigma = np.sqrt(self.constant * self.mode.value * sigma)
            criterion = self.mode.value - sigma
            tree_change = False
            remaining_node = []
            for node in self.partition:
                if node.value < criterion:
                    self.trash.add_child(node)
                    tree_change = True
                else:
                    remaining_node.append(node)
            if tree_change:
                self.update_trash()
                self.partition = remaining_node
                remaining_node.append(self.trash)
                root = Tree.huffman_build(remaining_node)
                self.replace_root(root)
                self.update_huffman_list()

    def _partition_rebalancing(self, epsilon: float):
        """
        Rebalance partition

        Merge as much sets as possible while ensuring for non-singleton set `S`
        .. math::`p(S) < \\max_y p(y) - \\epsilon`

        Parameters
        ----------
        epsilon:
        """
        # if the partition mode is a not singleton, refine the partition
        if not hasattr(self.mode, "label"):
            self._splitting(epsilon)
            self._merging(epsilon)
        # or if there exists a coarser partition, find it
        else:
            min1 = self.huffman_list[self._i_part].value
            min2 = self.huffman_list[self._i_part + 1].value
            criterion = self.mode.value / 2 - epsilon * self.root.value
            if min1 + min2 < criterion:
                self._merging(epsilon)

    def _splitting(self, epsilon: float):
        """
        Split elements of the current partition
        """
        codes = self.get_codes()
        setattr(self, "y_codes", codes[self.y_cat[:self.root.value]])
        self.get_leaves_set(self.root)
        self._refine_partition(epsilon)
        delattr(self, "y_codes")

    def _merging(self, epsilon: float):
        """
        Merge elements of the current partition
        """
        # we run Huffman at the partition level
        root = self.huffman_build(self.partition)
        self.replace_root(root)
        self.huffman_update()

        # update the partition, using the node value
        assert not hasattr(self, "y_codes")
        self._refine_partition(epsilon)

    def _refine_partition(self, epsilon: float):
        """
        Find current :math:`\eta`-admissible partition in the tree

        Here, :math:`\eta = \max N(y) / 2 n - \epsilon`
        """
        if hasattr(self, "y_codes"):
            # we are splitting nodes
            self.root.ind = np.ones(self.root.value, dtype=bool)
            n_mode = None
        else:
            # we are merging nodes
            n_mode = self.mode.value
        n = self.root.value

        # update the partition
        heap = []
        heapq.heappush(heap, (-self.root.value, self.root))
        self.partition = []
        while len(heap) > 0:
            # get the node with the biggest value
            _, node = heapq.heappop(heap)
            # if have reached our stop criterion, we stop
            if n_mode is not None and node.value < n_mode / 2 - epsilon * n:
                self.partition.append(node)
                break
            # the first leaf we find if the mode
            if isinstance(node, Leaf):
                self.partition.append(node)
                if n_mode is None:
                    self.mode = node
                    n_mode = node.value
                continue
            if hasattr(self, "y_codes"):
                # make n_node queries to get children information
                self._report_count(node, self.y_codes)
            # push children in the heap
            # be careful that we need to inverse the order
            loffset = {False: 0, True: 0.5}[type(node.left) is Leaf]
            roffset = {False: 0, True: 0.5}[type(node.right) is Leaf]
            heapq.heappush(heap, (-node.left.value + loffset, node.left))
            heapq.heappush(heap, (-node.right.value + roffset, node.right))
        # the remaning node form the partition
        while len(heap) > 0:
            _, node = heapq.heappop(heap)
            self.partition.append(node)

        # update the minimum partition node index
        self.partition_update()

    def _report_count(self, node, y_codes):
        """
        Query observation to refine information at node level

        Parameters
        ----------
        node: Node
            Node to refine observations
        y_codes: numpy.ndarray of int of size (n, c)
            Matrix of codes for each observed class
        """
        if node.value == 0:
            return

        # get child dispenser
        pos_ind = y_codes[node.ind, node.depth] == 1

        node.right.ind = node.ind.copy()
        node.right.ind[node.ind] = pos_ind
        node.right.value = np.sum(pos_ind)

        node.left.ind = node.ind.copy()
        node.left.ind[node.ind] = ~pos_ind
        node.left.value = node.value - node.right.value

        rcode, lcode = node.right._set_code, node.left._set_code

        # only queries for new information
        y_obs = self.y_observations[:self.root.value]
        right_unknown = y_obs[node.right.ind] & ~rcode
        right_queries = (right_unknown.sum(axis=1) != 0).sum()
        left_unknown = y_obs[node.left.ind] & ~lcode
        left_queries = (left_unknown.sum(axis=1) != 0).sum()
        self.nb_queries += right_queries + left_queries

        # update observations
        self.y_observations[:self.root.value][node.right.ind] &= rcode
        self.y_observations[:self.root.value][node.left.ind] &= lcode

    def _vitter_update(self, node):
        """
        Update Huffman tree due to recent observation

        This method assimilates to dynamic Huffman coding (Vitter algorithm)

        Parameters
        ----------
        node: Node
            Node to update
        """
        # stop at the root node
        if node.parent is None:
            node.value += 1
            return

        # get the node and swap it with max equal elements
        i_node = node._i_huff
        i_swap = i_node
        while not node < self.huffman_list[i_swap + 1]:
            i_swap += 1
        if i_swap != i_node:
            self._swap(i_node, i_swap)
            i_node = i_swap

        # technicalities related to new nodes
        if type(node) is Leaf and node.value == 0:
            self._nyo_update(node)
            return

        # increase node value
        node.value += 1
        parent = node.parent

        # keep the huffman order by swapping the node again
        node.parent.value += 1
        i_swap = i_node
        while self.huffman_list[i_swap + 1] < node:
            i_swap += 1
        node.parent.value -= 1
        if i_swap != i_node:
            swapped = self.huffman_list[i_swap]
            self._swap(i_node, i_swap)

            # update the parent whose value has changed
            if swapped.value != node.value:
                parent = node.parent

        # update the parent node
        self._vitter_update(parent)

    def _nyo_update(self, leaf):
        """
        Dealing with new observations
        """
        # find parent with no observation
        node = leaf
        while node.parent is not None and node.parent.value == 0:
            node = node.parent

        leaves_set = set(self.get_leaves_set(node))
        leaves_set.remove(leaf.label)
        if len(leaves_set) > 1:
            grand_parent = node.parent
            # create new `not yet observed`` balanced tree
            nyo_leaves = [self.y2leaf[i] for i in leaves_set]
            nyo = Tree.build_balanced_subtree(nyo_leaves)
            # slide newly observed leaf up in the tree
            new_node = Node(nyo, leaf)
            if grand_parent is None:
                self.replace_root(new_node)
            else:
                new_node.parent = grand_parent
                if grand_parent.left == node:
                    grand_parent.left = new_node
                elif grand_parent.right == node:
                    grand_parent.right = new_node
                new_node.update_depth(node.depth)
                del node

            # update attributes
            self.huffman_list = self.get_huffman_list()
            for i, node in enumerate(self.huffman_list):
                node._i_huff = i
            self.partition_update()

        leaf.value += 1
        self._vitter_update(leaf.parent)

    def _swap(self, i_node, i_swap):
        """
        Swap two nodes in the Huffman tree.
        """
        node = self.huffman_list[i_node]
        swapped = self.huffman_list[i_swap]

        # update list and pointers
        self.huffman_list[i_node] = swapped
        swapped._i_huff = i_node
        self.huffman_list[i_swap] = node
        node._i_huff = i_swap

        # swap in the tree
        self.swap(node, swapped)

    def huffman_update(self):
        """
        Update model and nodes attributes to remember Huffman positions.
        """
        self.huffman_list = self.get_huffman_list()
        for i, node in enumerate(self.huffman_list):
            node._i_huff = i

    def partition_update(self):
        """
        Update partition dictionary and set codes.
        """
        # update the partition dictionary
        self._i_part = np.inf
        for node in self.partition:
            y_set = self.get_leaves_set(node)
            for y in y_set:
                self.y2node[y] = node
            if self._i_part > node._i_huff:
                self._i_part = node._i_huff

    def get_leaves_set(self, node):
        """
        Get list of leaf descendants labels and set codes.
        """
        if type(node) is Leaf:
            y_set = [node.label]
        else:
            left_set = self.get_leaves_set(node.left)
            right_set = self.get_leaves_set(node.right)
            y_set = left_set + right_set
        node._set_code = np.zeros(self.m, dtype=bool)
        node._set_code[y_set] = 1
        return y_set

    def update_trash(self):
        """
        Get list of leaf descendants labels and set codes.
        """
        y_set = []
        for child in self.trash.children:
            y_set += self.get_leaves_set(child)
        self.trash._set_code = np.zeros(self.m, dtype=bool)
        self.trash._set_code[y_set] = 1
        for y in y_set:
            self.y2node[y] = self.trash

    def __repr__(self):
        return f"SetElimination at {id(self)}"
