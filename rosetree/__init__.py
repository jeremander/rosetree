"""Library implementing the "rose tree" data structure."""

from .draw import TreeDrawOptions, TreeLayoutOptions
from .tree import FrozenTree, MemoTree, Tree, zip_trees, zip_trees_with
from .trie import Trie


__version__ = '0.1.0'

__all__ = [
    'FrozenTree',
    'MemoTree',
    'Tree',
    'TreeDrawOptions',
    'TreeLayoutOptions',
    'Trie',
    'zip_trees',
    'zip_trees_with',
]
