import pytest

from rosetree import Tree


Tr = Tree

TREE1 = Tr(0, [Tr(1), Tr(2, [Tr(3, [Tr(4, [Tr(5)])]), Tr(6, [Tr(7), Tr(8)])]), Tr(9)])

TREE1_PRETTY = {
    'long': """
0
├── 1
├── 2
│   ├── 3
│   │   └── 4
│   │       └── 5
│   └── 6
│       ├── 7
│       └── 8
└── 9
"""
}

@pytest.mark.parametrize(['tree', 'style', 'output'], [
    pytest.param(TREE1, 'long', TREE1_PRETTY['long'], id='tree1_long'),
])
def test_pretty(tree, style, output):
    """Tests pretty string drawing of trees."""
    assert tree.pretty(style=style).strip() == output.strip()
