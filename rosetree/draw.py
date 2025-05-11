"""This module contains algorithms for drawing trees prettily."""

from collections.abc import Sequence
from typing import TypeVar

from .tree import BaseTree


T = TypeVar('T')


def _get_pretty_long(tree: BaseTree[T]) -> str:
    """Given a tree whose nows can be converted to strings via `str`, produces a pretty rendering of that tree in "long format."
    Each node is printed on its own line.
    This format is analogous to the Linux `tree` command."""
    def _pretty_lines(node: T, children: Sequence[list[str]]) -> list[str]:
        lines = [str(node)]
        num_children = len(children)
        for (i, child_lines) in enumerate(children):
            assert child_lines
            if i < num_children - 1:
                prefix1 = '├── '
                prefix2 = '│   '
            else:
                prefix1 = '└── '
                prefix2 = '    '
            lines.append(prefix1 + child_lines[0])
            lines.extend([prefix2 + line for line in child_lines[1:]])
        return lines
    lines = tree.fold(_pretty_lines)
    return '\n'.join(lines)
