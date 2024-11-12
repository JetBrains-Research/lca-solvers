import math


class SegmentTree:
    def __init__(self, size: int) -> None:
        self.size = size
        num_nodes = 2 ** (math.ceil(math.log2(size)) + 1)
        self.num_internal_nodes = num_nodes // 2

        # zero node is dummy and exists for indexing shift
        self.node_weights = [0 for _ in range(num_nodes)]
        self.insert(size)

    @staticmethod
    def children(node_idx: int) -> tuple[int, int]:
        return 2 * node_idx, 2 * node_idx + 1

    @staticmethod
    def parent(node_idx: int) -> int:
        return node_idx // 2

    def is_leaf(self, node_idx: int) -> bool:
        return node_idx >= self.num_internal_nodes

    def leaf_to_node_index(self, leaf_idx: int) -> int:
        return self.num_internal_nodes + leaf_idx - 1

    def node_to_leaf_index(self, node_idx: int) -> int:
        return node_idx - self.num_internal_nodes + 1

    def __str__(self) -> str:
        return str(self.node_weights[self.num_internal_nodes:(self.num_internal_nodes + self.size)])

    def __repr__(self) -> str:
        return f'Internal Nodes: {self.node_weights[1:self.num_internal_nodes]}\nLeaves: {self}'

    def __getitem__(self, leaf_idx: int) -> int:
        return self.node_weights[self.leaf_to_node_index(leaf_idx)]

    def find_best_fit_capacity(self, query: int) -> int:
        node_idx = 1

        while not self.is_leaf(node_idx):
            left_node_idx, right_node_idx = self.children(node_idx)
            if self.node_weights[left_node_idx] >= query:
                node_idx = left_node_idx
            else:
                node_idx = right_node_idx

        return self.node_to_leaf_index(node_idx)

    def update(self, node_idx: int) -> None:
        while node_idx != 1:
            node_idx = self.parent(node_idx)
            left_node_idx, right_node_idx = self.children(node_idx)
            self.node_weights[node_idx] = max(self.node_weights[left_node_idx], self.node_weights[right_node_idx])

    def insert(self, weight: int) -> None:
        if weight > self.size:
            raise IndexError(f'Trying to insert a weight that is greater than tree size: {weight} > {self.size}.')

        node_idx = self.leaf_to_node_index(weight)
        if self.node_weights[node_idx]:
            raise RuntimeError('Leaf is already full.')

        self.node_weights[node_idx] = weight
        self.update(node_idx)

    def delete(self, weight: int) -> None:
        node_idx = self.leaf_to_node_index(weight)
        if not self.node_weights[node_idx]:
            raise RuntimeError('Leaf is already empty.')

        self.node_weights[node_idx] = 0
        self.update(node_idx)


def main() -> None:
    # appendix example
    tree = SegmentTree(8)
    print(repr(tree))

    tree.insert(4)
    print(repr(tree))
    tree.insert(2)
    print(repr(tree))

    print(tree.find_best_fit_capacity(3))

    tree.delete(4)
    print(repr(tree))
    tree.insert(1)
    print(repr(tree))
