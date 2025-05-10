"""This module provides features for *weighted trees*."""

from collections.abc import Sequence
from math import isfinite
from operator import add, itemgetter
from typing import NamedTuple, Optional, Type, TypeVar, cast

from .tree import BaseTree, zip_trees_with


T = TypeVar('T')


# nonnegative number
Weight = float
# node value paired with a weight
WeightedNode = tuple[Weight, T]
# a tree with weights on the nodes
NodeWeightedTree = BaseTree[WeightedNode[T]]


def _safe_divide(weight1: Weight, weight2: Weight) -> Optional[float]:
    """Division where 0/0 is None."""
    if (weight1 == weight2) and (weight1 == 0.0):
        return None
    return weight1 / weight2


class NodeWeightInfo(NamedTuple):
    """Tuple type containing the following info about a node in a node-weighted tree:
        - `weight`: the node's own (local) weight
        - `subtotal`: the node's "subtotal" (total weight of the node's subtree, including itself)
        - `self_to_subtotal`: ratio of the node's weight to its subtotal
        - `self_to_global`: ratio of the node's weight to the global total
        - `subtotal_to_parent`: ratio of the node's subtotal to its parent's subtotal (1 if this node is the root)
        - `subtotal_to_global`: ratio of the node's subtotal to the global total
    It is assumed that all weights are nonnegative.
    If a ratio is 0/0, the value will be None."""
    weight: float
    subtotal: float
    self_to_subtotal: Optional[float]
    self_to_global: Optional[float]
    subtotal_to_parent: Optional[float]
    subtotal_to_global: Optional[float]


def aggregate_weight_info(tree: BaseTree[Weight]) -> BaseTree[NodeWeightInfo]:
    """Given a tree of (node-local) weights, aggregates this into an identically structured tree of `NodeWeightInfo` providing more information such as subtree total weight, fraction of a subtree's total to its parent subtree's total, etc.
    Raises a ValueError if any weight is negative."""
    cls = cast(Type[BaseTree[NodeWeightInfo]], type(tree))
    global_total = tree.reduce(add)
    def check_valid_weight(weight: Weight) -> None:
        if not isfinite(weight):
            raise ValueError(f'encountered weight {weight}, all weights must be finite')
        if weight < 0.0:
            raise ValueError(f'encountered weight {weight}, all weights must be nonnegative')
    def func(node: Weight, children: Sequence[BaseTree[NodeWeightInfo]]) -> BaseTree[NodeWeightInfo]:
        check_valid_weight(node)
        subtotal = node + sum(child.node.subtotal for child in children)
        self_to_subtotal = _safe_divide(node, subtotal)
        self_to_global = _safe_divide(node, global_total)
        subtotal_to_global = _safe_divide(subtotal, global_total)
        # temporary value for subtotal_to_parent (will get replaced when processing parent)
        placeholder = None if (subtotal == 0.0) else 1.0
        info = NodeWeightInfo(node, subtotal, self_to_subtotal, self_to_global, placeholder, subtotal_to_global)
        # modify childrens' subtotal_to_parent
        def fix_child(child: BaseTree[NodeWeightInfo]) -> BaseTree[NodeWeightInfo]:
            subtotal_to_parent = _safe_divide(child.node.subtotal, subtotal)
            child_node = child.node._replace(subtotal_to_parent=subtotal_to_parent)
            return cls(child_node, list(child))
        return cls(info, list(map(fix_child, children)))
    return tree.fold(func)

def aggregate_node_weighted_tree(tree: NodeWeightedTree[T]) -> BaseTree[tuple[NodeWeightInfo, T]]:
    """Given a weighted tree whose nodes are (weight, data) pairs, returns an identically structured tree where instead of weights we have `NodeWeightInfo` tuples providing more information such as subtree total weight, fraction of a subtree's total to its parent subtree's total, etc.
    Raises a ValueError if any weight is negative."""
    weight_info_tree = aggregate_weight_info(tree.map(itemgetter(0)))
    def replace_weight(info: NodeWeightInfo, pair: WeightedNode[T]) -> tuple[NodeWeightInfo, T]:
        return (info, pair[1])
    return zip_trees_with(replace_weight, weight_info_tree, tree)  # type: ignore[misc]
