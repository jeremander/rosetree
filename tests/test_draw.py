import importlib
from operator import itemgetter

import matplotlib
import pytest

from rosetree import Tree


Tr = Tree

PRETTY_TREES = [
    {
        'tree': Tr(0),
        'bottom-up': ' 0 ',
        'top-down': ' 0 ',
        'long': '0',  # TODO: should this be distinguished somehow?
    },
    {
        'tree': Tr(0, [Tr(1)]),
        'bottom-up': """
 0
 │
 1
 """,
        'top-down': """
 0
 │
 1
 """ ,
        'long': """
0
└── 1
"""
    },
    {
        'tree': Tr(10, [Tr(1)]),
        'bottom-up': """
 10
 │
 1
 """,
        'top-down': """
 10
 │
 1
 """ ,
        'long': """
10
└── 1
"""
    },
    {
        'tree': Tr(0, [Tr(1, [Tr(2)])]),
        'bottom-up': """
 0
 │
 1
 │
 2
 """,
        'top-down': """
 0
 │
 1
 │
 2
 """ ,
        'long': """
0
└── 1
    └── 2
"""
    },
    {
        'tree': Tr(0, [Tr(1), Tr(2)]),
        'bottom-up': """
  0
 ┌┴─┐
 1  2
""",
        'top-down': """
  0
 ┌┴─┐
 1  2
""",
        'long': """
0
├── 1
└── 2
"""
    },
    {
        'tree': Tr(0, [Tr(1), Tr(2222222)]),
        'bottom-up': """
    0
 ┌──┴──┐
 1  2222222
""",
        'top-down': """
    0
 ┌──┴──┐
 1  2222222
""",
        'long': """
0
├── 1
└── 2222222
"""
    },
    {
        'tree': Tr(123456789, [Tr(1), Tr(2)]),
        'bottom-up': """
 123456789
 ┌───┴───┐
 1       2
""",
        'top-down': """
 123456789
 ┌───┴───┐
 1       2
""",
        'long': """
123456789
├── 1
└── 2
"""
    },
    {
        'tree': Tr(0, [Tr(1), Tr(2, [Tr(3)])]),
        'bottom-up': """
  0
 ┌┴─┐
 │  2
 │  │
 1  3
""",
        'top-down': """
  0
 ┌┴─┐
 1  2
    │
    3
""",
        'long': """
0
├── 1
└── 2
    └── 3
"""
    },
    {
        'tree': Tr(0, [Tr(1, [Tr(2)]), Tr(3)]),
        'bottom-up': """
  0
 ┌┴─┐
 1  │
 │  │
 2  3
""",
        'top-down': """
  0
 ┌┴─┐
 1  3
 │
 2
""",
        'long': """
0
├── 1
│   └── 2
└── 3
"""
    },
    {
        'tree': Tr(0, [Tr(1), Tr(2), Tr(3)]),
        'bottom-up': """
    0
 ┌──┼──┐
 1  2  3
""",
        'top-down': """
    0
 ┌──┼──┐
 1  2  3
""",
        'long': """
0
├── 1
├── 2
└── 3
"""
    },
    {
        'tree': Tr(0, [Tr(1, [Tr(2)]), Tr(3), Tr(4, [Tr(5, [Tr(6)])])]),
        'bottom-up': """
    0
 ┌──┼──┐
 │  │  4
 │  │  │
 1  │  5
 │  │  │
 2  3  6
""",
        'top-down': """
    0
 ┌──┼──┐
 1  3  4
 │     │
 2     5
       │
       6
""",
        'long': """
0
├── 1
│   └── 2
├── 3
└── 4
    └── 5
        └── 6
"""
    },
    {
        'tree': Tr(0, [Tr(1), Tr(2, [Tr(3, [Tr(4, [Tr(5)])]), Tr(6, [Tr(7), Tr(8)]), Tr(9)]), Tr(10)]),
        'bottom-up': """
        0
 ┌──────┼───────┐
 │      2       │
 │  ┌───┼────┐  │
 │  3   │    │  │
 │  │   │    │  │
 │  4   6    │  │
 │  │  ┌┴─┐  │  │
 1  5  7  8  9  10
""",
        'top-down': """
        0
 ┌──────┼───────┐
 1      2       10
    ┌───┼────┐
    3   6    9
    │  ┌┴─┐
    4  7  8
    │
    5
""",
        'long': """
0
├── 1
├── 2
│   ├── 3
│   │   └── 4
│   │       └── 5
│   ├── 6
│   │   ├── 7
│   │   └── 8
│   └── 9
└── 10
""",
    },
    {
        'tree': Tr(''),
        'bottom-up': '',
        'top-down': '',
        'long': '',
    },
    {
        'tree': Tr('', [Tree(''), Tree(2)]),
        'bottom-up': """

 ┌┴┐
   2
""",
        'top-down': """

 ┌┴┐
   2
""",
        'long': """

├──
└── 2
"""
    },
]

def _normalize_pretty(s):
    return '\n'.join(line.rstrip() for line in s.splitlines()).rstrip('\n')

def _is_increasing(it, strict: bool = True):
    prev = None
    for elt in it:
        if (prev is not None) and ((elt <= prev) if strict else (elt < prev)):
            return False
        prev = elt
    return True

@pytest.fixture
def use_agg_backend():
    """Temporarily sets the matplotlib backend to 'Agg', which does not make use of a GUI."""
    backend = matplotlib.get_backend()
    matplotlib.use('Agg', force=True)
    importlib.reload(matplotlib.pyplot)
    yield
    matplotlib.use(backend, force=True)
    importlib.reload(matplotlib.pyplot)

@pytest.mark.parametrize('item', PRETTY_TREES)
def test_pretty(item):
    """Tests pretty string drawing of trees."""
    tree = item['tree']
    for style in ['bottom-up', 'top-down', 'long']:
        if style in item:
            output = item[style]
            if output.startswith('\n'):
                output = output[1:]
            assert _normalize_pretty(tree.pretty(style=style)) == _normalize_pretty(output)

def test_invalid_pretty_style():
    """Tests that a ValueError is raised if an invalid pretty style is provided to Tree.pretty."""
    with pytest.raises(ValueError, match="invalid pretty tree style 'fake'"):
        _ = PRETTY_TREES[0]['tree'].pretty(style='fake')

@pytest.mark.parametrize('tree', [item['tree'] for item in PRETTY_TREES])
@pytest.mark.parametrize('style', ['top-down', 'bottom-up'])
def test_with_bounding_boxes(tree, style):
    """Tests properties of Tree.with_bounding_boxes."""
    # get number of lines of each node in the tree
    num_lines_tree = tree.map(lambda node: len(str(node).splitlines()))
    min_num_lines = num_lines_tree.reduce(min)
    max_num_lines = num_lines_tree.reduce(max)
    box_tree = tree.with_bounding_boxes(style=style).map(itemgetter(0))
    if style == 'top-down':
        groups = list(box_tree.depth_sorted_nodes())
    else:
        groups = list(box_tree.height_sorted_nodes())
    # y is the same for all nodes of the same depth or height,
    # and x is strictly increasing for each layer (assuming pre-order or post-order traversal)
    for group in groups:
        if min_num_lines == max_num_lines == 1:
            # node y values may vary based on lines of text, so only check this if all nodes are 1 line
            assert len({box.y for (box, _) in group}) == 1
            assert len({box.y for (_, box) in group}) == 1
        assert _is_increasing([box.x for (box, _) in group])
        assert _is_increasing([box.x for (_, box) in group])
    if style == 'top-down':
        # y must increase with decreasing depth
        assert _is_increasing([-group[0][0].y for group in groups])
        assert _is_increasing([-group[0][1].y for group in groups])
    else:
        # y must increase with increasing height
        assert _is_increasing([group[0][0].y for group in groups])
        assert _is_increasing([group[0][1].y for group in groups])
    # left-to-right leaves must have increasing x
    boxes = list(box_tree.iter_leaves())
    assert _is_increasing([box.x for (box, _) in boxes])
    assert _is_increasing([box.x for (_, box) in boxes])

DRAW_TREES = [PRETTY_TREES[i] for i in [1, 2, 12]]

@pytest.mark.parametrize('tree', [item['tree'] for item in DRAW_TREES])
@pytest.mark.parametrize('style', ['top-down', 'bottom-up'])
def test_draw_matplotlib(use_agg_backend, monkeypatch, tree, style):
    """Tests running Tree.draw.
    Does not launch a matplotlib GUI, just checks that the command runs without error."""
    monkeypatch.setattr(matplotlib.pyplot, 'show', lambda: None)
    tree.draw(style=style)
