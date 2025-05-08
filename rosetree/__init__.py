"""Library implementing the "rose tree" data structure."""

from .tree import FrozenTree, MemoTree, Tree
from .trie import Trie


__version__ = '0.1.0'

__all__ = [
    'FrozenTree',
    'MemoTree',
    'Tree',
    'Trie',
]
