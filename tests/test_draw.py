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
