from math import inf
import operator
from operator import add, attrgetter

import pytest

from rosetree import Tree
from rosetree.weighted import NodeWeightInfo, Treemap, aggregate_weight_info


def flip(pair):
    return pair[::-1]

@pytest.mark.parametrize(['tree', 'err_type', 'err_match'], [
    (
        Tree(-1),
        ValueError,
        'encountered weight -1, all weights must be nonnegative'
    ),
    (
        Tree(-inf),
        ValueError,
        'encountered weight -inf, all weights must be finite'
    ),
    (
        Tree(inf),
        ValueError,
        'encountered weight inf, all weights must be finite'
    ),
    (
        Tree(0, [Tree(-1)]),
        ValueError,
        'encountered weight -1, all weights must be nonnegative'
    ),
    (
        Tree(0, [Tree(-1), Tree(-2)]),
        ValueError,
        'encountered weight -1, all weights must be nonnegative'
    ),
    # validation is depth-first
    (
        Tree(0, [Tree(0, [Tree(-1)]), Tree(-2)]),
        ValueError,
        'encountered weight -1, all weights must be nonnegative'
    ),
    (
        Tree('a'),
        TypeError,
        'must be real number, not str'
    ),
    (
        Tree(None),
        TypeError,
        'must be real number, not NoneType'
    ),
])
def test_aggregate_weight_info_invalid(tree, err_type, err_match):
    with pytest.raises(err_type, match=err_match):
        _ = aggregate_weight_info(tree)

@pytest.mark.parametrize(['tree1', 'tree2'], [
    (
        Tree(0),
        Tree(NodeWeightInfo(0, 0, None, None, None, None))
    ),
    (
        Tree(1),
        Tree(NodeWeightInfo(1, 1, 1, 1, 1, 1))
    ),
    (
        Tree(2),
        Tree(NodeWeightInfo(2, 2, 1, 1, 1, 1))
    ),
    (
        Tree(0, [Tree(0)]),
        Tree(NodeWeightInfo(0, 0, None, None, None, None), [Tree(NodeWeightInfo(0, 0, None, None, None, None))])
    ),
    (
        Tree(0, [Tree(1)]),
        Tree(NodeWeightInfo(0, 1, 0, 0, 1, 1), [Tree(NodeWeightInfo(1, 1, 1, 1, 1, 1))])
    ),
    (
        Tree(1, [Tree(0)]),
        Tree(NodeWeightInfo(1, 1, 1, 1, 1, 1), [Tree(NodeWeightInfo(0, 0, None, 0, 0, 0))])
    ),
    (
        Tree(1, [Tree(1)]),
        Tree(NodeWeightInfo(1, 2, 1/2, 1/2, 1, 1), [Tree(NodeWeightInfo(1, 1, 1, 1/2, 1/2, 1/2))])
    ),
    (
        Tree(1, [Tree(3)]),
        Tree(NodeWeightInfo(1, 4, 1/4, 1/4, 1, 1), [Tree(NodeWeightInfo(3, 3, 1, 3/4, 3/4, 3/4))])
    ),
    (
        Tree(0, [Tree(0), Tree(0)]),
        Tree(NodeWeightInfo(0, 0, None, None, None, None), [Tree(NodeWeightInfo(0, 0, None, None, None, None)), Tree(NodeWeightInfo(0, 0, None, None, None, None))])
    ),
    (
        Tree(0, [Tree(0), Tree(1)]),
        Tree(NodeWeightInfo(0, 1, 0, 0, 1, 1), [Tree(NodeWeightInfo(0, 0, None, 0, 0, 0)), Tree(NodeWeightInfo(1, 1, 1, 1, 1, 1))])
    ),
    (
        Tree(1, [Tree(0), Tree(0)]),
        Tree(NodeWeightInfo(1, 1, 1, 1, 1, 1), [Tree(NodeWeightInfo(0, 0, None, 0, 0, 0)), Tree(NodeWeightInfo(0, 0, None, 0, 0, 0))])
    ),
    (
        Tree(0, [Tree(1), Tree(3)]),
        Tree(NodeWeightInfo(0, 4, 0, 0, 1, 1), [Tree(NodeWeightInfo(1, 1, 1, 1/4, 1/4, 1/4)), Tree(NodeWeightInfo(3, 3, 1, 3/4, 3/4, 3/4))])
    ),
    (
        Tree(1, [Tree(1), Tree(3)]),
        Tree(NodeWeightInfo(1, 5, 1/5, 1/5, 1, 1), [Tree(NodeWeightInfo(1, 1, 1, 1/5, 1/5, 1/5)), Tree(NodeWeightInfo(3, 3, 1, 3/5, 3/5, 3/5))])
    ),
    (
        Tree(0, [Tree(0, [Tree(0)])]),
        Tree(NodeWeightInfo(0, 0, None, None, None, None), [Tree(NodeWeightInfo(0, 0, None, None, None, None), [Tree(NodeWeightInfo(0, 0, None, None, None, None))])])
    ),
    (
        Tree(0, [Tree(0, [Tree(1)])]),
        Tree(NodeWeightInfo(0, 1, 0, 0, 1, 1), [Tree(NodeWeightInfo(0, 1, 0, 0, 1, 1), [Tree(NodeWeightInfo(1, 1, 1, 1, 1, 1))])])
    ),
    (
        Tree(0, [Tree(1, [Tree(0)])]),
        Tree(NodeWeightInfo(0, 1, 0, 0, 1, 1), [Tree(NodeWeightInfo(1, 1, 1, 1, 1, 1), [Tree(NodeWeightInfo(0, 0, None, 0, 0, 0))])])
    ),
    (
        Tree(1, [Tree(0, [Tree(0)])]),
        Tree(NodeWeightInfo(1, 1, 1, 1, 1, 1), [Tree(NodeWeightInfo(0, 0, None, 0, 0, 0), [Tree(NodeWeightInfo(0, 0, None, 0, None, 0))])])
    ),
    (
        Tree(1, [Tree(1, [Tree(2)])]),
        Tree(NodeWeightInfo(1, 4, 1/4, 1/4, 1, 1), [Tree(NodeWeightInfo(1, 3, 1/3, 1/4, 3/4, 3/4), [Tree(NodeWeightInfo(2, 2, 1, 1/2, 2/3, 1/2))])])
    ),
    (
        Tree(0, [Tree(0, [Tree(1), Tree(2)]), Tree(0, [Tree(3)])]),
        Tree(NodeWeightInfo(0, 6, 0, 0, 1, 1), [Tree(NodeWeightInfo(0, 3, 0, 0, 1/2, 1/2), [Tree(NodeWeightInfo(1, 1, 1, 1/6, 1/3, 1/6)), Tree(NodeWeightInfo(2, 2, 1, 1/3, 2/3, 1/3))]), Tree(NodeWeightInfo(0, 3, 0, 0, 1/2, 1/2), [Tree(NodeWeightInfo(3, 3, 1, 1/2, 1, 1/2))])])
    ),
    # (weight, subtotal, self_to_subtotal, self_to_global, subtotal_to_parent, subtotal_to_global)
])
def test_aggregate_weight_info_valid(tree1, tree2):
    tree1_agg = aggregate_weight_info(tree1)
    # weights match their original ones
    assert tree1_agg.map(attrgetter('weight')) == tree1
    # subtotals match those calculated via scan
    assert tree1_agg.map(attrgetter('subtotal')) == tree1.scan(add)
    # self_to_subtotal is weight/subtotal
    def is_valid(info):
        if info.subtotal == 0.0:
            return info.self_to_subtotal is None
        return info.self_to_subtotal == info.weight / info.subtotal
    assert tree1_agg.map(is_valid).reduce(operator.and_)
    # result matches what we expect
    assert tree1_agg == tree2
    # create tree of (weight, index) tuples
    tree3 = tree1.tag_with_unique_counter().map(flip)
    tree4 = tree1_agg.tag_with_unique_counter().map(flip)
    # aggregation of tree with counter tags matches counter-tagged aggregate tree
    assert Treemap.from_node_weighted_tree(tree3) == Treemap.wrap(tree4, deep=True)
