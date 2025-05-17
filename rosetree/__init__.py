"""Library implementing the "rose tree" data structure."""

from .draw import TreeDrawOptions, TreeLayoutOptions
from .tree import FrozenTree, MemoTree, Tree, zip_trees, zip_trees_with
from .trie import Trie
from .weighted import Treemap


__version__ = '0.2.0'

__all__ = [
    'FrozenTree',
    'MemoTree',
    'Tree',
    'TreeDrawOptions',
    'TreeLayoutOptions',
    'Treemap',
    'Trie',
    'zip_trees',
    'zip_trees_with',
]
