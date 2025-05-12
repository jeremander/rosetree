import pytest

from rosetree import Tree


Tr = Tree

TREE1 = Tr(0, [Tr(1), Tr(2, [Tr(3, [Tr(4, [Tr(5)])]), Tr(6, [Tr(7), Tr(8)]), Tr(9)]), Tr(10)])

TREE1_PRETTY = {
    'bottom-up': """
          0
 ┌────────┴┬────────┐
 │         │        │
 │         2        │
 │    ┌────┴┬────┐  │
 │    │     │    │  │
 │    3     │    │  │
 │    │     6    │  │
 │    4    ┌┴─┐  │  │
 │    │    │  │  │  │
 1    5    7  8  9  10
""",
    'top-down': """
          0
 ┌────────┴┬────────┐
 │         │        │
 1         2        10
      ┌────┴┬────┐
      │     │    │
      3     6    9
      │    ┌┴─┐
      4    │  │
      │    7  8
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
}

def _normalize_pretty(s):
    return '\n'.join(line.rstrip() for line in s.splitlines())

@pytest.mark.parametrize(['tree', 'style', 'output'], [
    pytest.param(TREE1, 'bottom-up', TREE1_PRETTY['bottom-up'], id='tree1_bottom_up'),
    pytest.param(TREE1, 'top-down', TREE1_PRETTY['top-down'], id='tree1_top_down'),
    pytest.param(TREE1, 'long', TREE1_PRETTY['long'], id='tree1_long'),
])
def test_pretty(tree, style, output):
    """Tests pretty string drawing of trees."""
    assert _normalize_pretty(tree.pretty(style=style)) == output.strip('\n')

def test_invalid_pretty_style():
    """Tests that a ValueError is raised if an invalid pretty style is provided to Tree.pretty."""
    with pytest.raises(ValueError, match="invalid pretty tree style 'fake'"):
        _ = TREE1.pretty(style='fake')
