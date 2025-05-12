"""This module contains algorithms for drawing trees prettily."""

from collections.abc import Sequence
import re
from typing import NamedTuple, Optional, TypeVar

from typing_extensions import Self

from .tree import BaseTree


T = TypeVar('T')


# LONG FORMAT

def pretty_tree_long(tree: BaseTree[T]) -> str:
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

# WIDE FORMAT

def _center_index(n: int) -> int:
    """Gets the (integer) midpoint index for the given distance n."""
    return max(0, n - 1) // 2

_PARTITION_REGEX = re.compile(r'^(\s*)([^\s](.*[^\s])?)(\s*)$')

def _partition_line(line: str) -> tuple[str, str, str]:
    """Giving a line with leading and/or trailing whitespace, splits it into these three parts, returning a tuple (leading whitespace, text, trailing whitespace)."""
    (leading, text, _, trailing) = _PARTITION_REGEX.match(line).groups()  # type: ignore[union-attr]
    return (leading, text, trailing)

def _place_line(line: str, width: int, center: int) -> str:
    """Given a text line, total width, and center index, pads the line on the left and right so that the total width and text center match the target quantities."""
    line_length = len(line)
    line_center = _center_index(line_length)
    lpad = center - line_center
    rpad = (width - center) - (line_length - line_center)
    if (lpad < 0) or (rpad < 0):
        raise ValueError(f'cannot center line of length {line_length} at index {center} in a width of {width}')
    return (' ' * lpad) + line + (' ' * rpad)

def _get_box_char(j: int, lcenter: int, midpoint: int, rcenter: int) -> str:
    if j == midpoint:
        return '┴'
    if (j < lcenter) or (j > rcenter):
        return ' '
    if j == lcenter:
        return '┌'
    if j == rcenter:
        return '┐'
    return '─'

def _extend_box_char_down(c: str) -> str:
    if c == '─':
        return '┬'
    if c == '┴':
        return '┼'
    return c


class Column(NamedTuple):
    """Tuple of elements for a fixed-width column of text, which may consist of multiple rows."""
    width: int  # width of column
    center: int  # center index
    rows: list[str]  # rows of text

    @classmethod
    def conjoin(cls, columns: Sequence[Self], spacing: int, top_down: bool) -> Self:
        """Conjoint multiple columns horizontally into one.
        An integer, spacing, specifies how many spaces to insert between each column.
        If top_down = True, aligns conjoined columns from the top, otherwise from the bottom."""
        (widths, _, cols) = zip(*columns)
        heights = [len(col) for col in cols]
        max_height = max(heights)
        empty_rows = [[' ' * width] * (max_height - height) for (width, height) in zip(widths, heights)]
        if top_down:
            cols = tuple(col + empty for (empty, col) in zip(empty_rows, cols))
        else:
            cols = tuple(empty + col for (empty, col) in zip(empty_rows, cols))
        if len(columns) == 1:  # pad the outside
            width = widths[0] + 2 * (spacing - 1)
            delim = ' ' * (spacing - 1)
            col = [delim + row + delim for row in cols[0]]
        else:
            width = sum(widths) + (spacing * (len(cols) - 1))
            delim = ' ' * spacing
            col = [delim.join(row) for row in zip(*cols)]
        return cls(width, _center_index(width), col)


def pretty_tree_wide(tree: BaseTree[T], *, top_down: bool = False) -> str:
    """Given a tree whose nows can be converted to strings via `str`, produces a pretty rendering of that tree in "wide format."
    This presents the tree's root node at the top, with branches cascading down.
    If top_down = True, positions nodes of the same depth on the same vertical level.
    Otherwise, positions leaf nodes on the same vertical level."""
    def conjoin_subtrees(node: T, children: Sequence[Column]) -> Column:
        node_str = str(node)
        node_lines = node_str.splitlines() if node_str else ['']
        node_width = max(map(len, node_lines))
        node_lines = [line.ljust(node_width) for line in node_lines]
        if (num_children := len(children)) == 0:
            return Column(node_width, _center_index(node_width), node_lines)
        (child_widths, child_centers, _) = zip(*children)
        spacing = 2
        while True:
            (width, center, text) = Column.conjoin(children, spacing, top_down)
            # place parent at the midpoint of the left and right childrens' centers
            if num_children == 1:
                spans = [(0, width)]
                centers = [child_centers[0] + (spacing - 1)]
            else:
                spans = [(0, child_widths[0])]
                for child_width in child_widths[1:]:
                    start = spans[-1][1]
                    spans.append((start + spacing, start + spacing + child_width))
                centers = [span[0] + child_center for (span, child_center) in zip(spans, child_centers)]
            (lcenter, rcenter) = (centers[0], centers[-1])
            midpoint = lcenter + _center_index(rcenter - lcenter + 1)
            try:
                node_lines = [_place_line(line, width, midpoint) for line in node_lines]
                break
            except ValueError:
                # parent is too wide for the children: add more space to children
                spacing += 1  # TODO: can we calculate the right value up-front?
        if num_children == 1:
            edges = ['│' if (j == midpoint) else ' ' for j in range(width)]
        else:
            edges = [_get_box_char(j, lcenter, midpoint, rcenter) for j in range(width)]
            text = [' ' * width] + text
        node_lines.append(''.join(edges))
        # get mapping from center indices to column spans
        def find_span(center: int) -> Optional[tuple[int, int]]:
            return next(((start, stop) for (start, stop) in spans if start <= center < stop), None)
        column_spans = {c: find_span(c) for c in centers}
        # insert '|' downward to each child
        text_cols = [list(col) for col in zip(*text)]
        extension_indices = set()  # indices at which to branch downward
        for (j, col) in enumerate(text_cols):
            if (span := column_spans.get(j, None)) is None:
                continue
            for (i, row) in enumerate(text):
                if row[span[0]:span[1]].isspace():
                    col[i] = '│'
                    if i == 0:
                        extension_indices.add(j)
                else:
                    break
        if extension_indices:
            # branch downward from horizontal lines
            node_lines[1] = ''.join([_extend_box_char_down(c) if (j in extension_indices) else c for (j, c) in enumerate(node_lines[1])])
        text = node_lines + [''.join(row) for row in zip(*text_cols)]
        top_line = node_lines[0]
        if top_line.isspace():
            new_center = center
        else:
            (leading, top_text, _) = _partition_line(node_lines[0])
            new_center = len(leading) + _center_index(len(top_text))
        return Column(width, new_center, text)
    lines = tree.fold(conjoin_subtrees).rows
    # ensure the tree is padded on the left & right
    if any(not line.startswith(' ') for line in lines):
        lines = [' ' + line for line in lines]
    if any(not line.endswith(' ') for line in lines):
        lines = [line + ' ' for line in lines]
    return '\n'.join(lines) + '\n'
