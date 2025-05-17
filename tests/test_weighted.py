from math import inf
import operator
from operator import add, attrgetter

import plotly.graph_objects as go
import pytest

from rosetree import Tree, Trie
from rosetree.draw import _plotly_treemap_args
from rosetree.weighted import NodeWeightInfo, Treemap, TreemapStyle, aggregate_weight_info


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


# example monthly budget by category
BUDGET = [
    ('Finance:Retirement', 250),
    ('Finance:Saving', 400),
    ('Food:Groceries', 600),
    ('Food:Restaurant', 150),
    ('Food:Restaurant:FastFood', 50),
    ('Health', 400),
    ('Home:Rent', 1400),
    ('Home:Tech:Cellphone', 90),
    ('Home:Tech:Internet', 60),
    ('Home:Tech:Streaming', 45),
    ('Home:Utilities', 250),
    ('Car:Gas', 120),
    ('Car:Insurance', 75),
    ('Car:Payment', 250),
    ('Car:Tolls', 15),
    ('Shopping', 115),
    ('Shopping:Clothes', 100),
    ('Shopping:Hobbies', 150),
]

@pytest.fixture(scope='module')
def budget_tree():
    """Gets an example budget represented as a WeightedNodeTree whose nodes are (amount, category) pairs."""
    path_dict = {tuple(category.split(':')): amount for (category, amount) in BUDGET}
    trie = Trie.from_sequences(path_dict)
    def get_weighted_node(pair):
        (_, path) = pair
        amount = path_dict.get(path, 0)
        node = path[-1] if path else None
        return (amount, node)
    return Tree.from_trie(trie).map(get_weighted_node)

def test_budget_treemap(monkeypatch, tmp_path, budget_tree):
    """Tests drawing a plotly treemap for the example budget treemap."""
    monkeypatch.setattr(go.Figure, 'show', lambda _: None)
    treemap = Treemap.from_node_weighted_tree(budget_tree)
    args = _plotly_treemap_args(treemap)
    assert args['branchvalues'] == 'total'
    assert args['values'] == [info.subtotal for (info, _) in treemap.iter_nodes()]
    assert args['marker_colors'] is None
    color_func = lambda _: 'green'
    args = _plotly_treemap_args(treemap, color_func=color_func)
    assert args['marker_colors'] == ['green' for _ in range(treemap.size)]
    # create plotly plot (but don't actually display it)
    for style in TreemapStyle.__args__:
        treemap.draw_treemap(style=style)
    with pytest.raises(ValueError, match="invalid treemap style 'fake'"):
        treemap.draw_treemap(style='fake')
    # save plot to an SVG file
    try:
        svg_path = tmp_path / 'treemap.svg'
        treemap.draw_treemap(svg_path)
        assert svg_path.exists()
    except ValueError as e:
        pytest.skip(f'Could not save treemap: {e}')
