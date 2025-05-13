"""This module contains algorithms for drawing trees prettily."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil
import re
from typing import TYPE_CHECKING, Callable, NamedTuple, Type, TypeVar, cast

from typing_extensions import Self

from .utils import cumsums


if TYPE_CHECKING:
    from .tree import BaseTree


T = TypeVar('T')

_PARTITION_REGEX = re.compile(r'^(\s*)([^\s](.*[^\s])?)(\s*)$')


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

def _pad_lines(lines: list[str]) -> list[str]:
    # ensure the tree is padded on the left & right
    if any(not line.startswith(' ') for line in lines):
        lines = [' ' + line for line in lines]
    if any(not line.endswith(' ') for line in lines):
        lines = [line + ' ' for line in lines]
    return lines

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
        assert len(columns) > 1
        (widths, _, cols) = zip(*columns)
        heights = [len(col) for col in cols]
        max_height = max(heights)
        empty_rows = [[' ' * width] * (max_height - height) for (width, height) in zip(widths, heights)]
        if top_down:
            cols = tuple(col + empty for (empty, col) in zip(empty_rows, cols))
        else:
            cols = tuple(empty + col for (empty, col) in zip(empty_rows, cols))
        width = sum(widths) + (spacing * (len(cols) - 1))
        delim = ' ' * spacing
        col = [delim.join(row) for row in zip(*cols)]
        return cls(width, _center_index(width), col)

    def pad_to(self, width: int) -> Self:
        """Pads the column to the given width."""
        n = width - self.width
        if n <= 0:
            return self
        lpad = ' ' * (n // 2)
        rpad = ' ' * (n - n // 2)
        rows = [lpad + row + rpad for row in self.rows]
        return type(self)(width, _center_index(width), rows)


def pretty_tree_wide(tree: BaseTree[T], *, top_down: bool = False, spacing: int = 2) -> str:
    """Given a tree whose nows can be converted to strings via `str`, produces a pretty rendering of that tree in "wide format."
    This presents the tree's root node at the top, with branches cascading down.
    If top_down = True, positions nodes of the same depth on the same vertical level.
    Otherwise, positions leaf nodes on the same vertical level.
    spacing is an integer indicating the minimum number of spaces between each column."""
    def conjoin_subtrees(node: T, children: Sequence[Column]) -> Column:
        node_str = str(node)
        node_lines = node_str.splitlines() if node_str else ['']
        node_width = max(map(len, node_lines))
        node_lines = [line.ljust(node_width) for line in node_lines]
        if (num_children := len(children)) == 0:  # leaf
            return Column(node_width, _center_index(node_width), node_lines)
        (child_widths, child_centers, _) = zip(*children)
        if num_children == 1:
            (width, center, rows) = children[0]
            spans = [(0, width)]
            centers = [center]
            midpoint = center
            edges = ['│' if (j == center) else ' ' for j in range(width)]
        else:
            # calculate the smallest spacing required for the child width to exceed the parent width
            num_spaces = max(spacing, ceil((node_width - sum(child_widths)) / (num_children - 1)))
            (width, center, rows) = Column.conjoin(children, num_spaces, top_down)
            # place parent at the midpoint of the leftmotst and rightmost child's centers
            spans = [(0, child_widths[0])]
            for child_width in child_widths[1:]:
                start = spans[-1][1]
                spans.append((start + num_spaces, start + num_spaces + child_width))
            centers = [start + child_center for ((start, _), child_center) in zip(spans, child_centers)]
            (lcenter, rcenter) = (centers[0], centers[-1])
            midpoint = lcenter + _center_index(rcenter - lcenter + 1)
            edges = [_get_box_char(j, lcenter, midpoint, rcenter) for j in range(width)]
            for j in centers:
                edges[j] = _extend_box_char_down(edges[j])
        node_lines = [_place_line(line, width, midpoint) for line in node_lines]
        node_lines.append(''.join(edges))
        # get mapping from center indices to column spans
        column_spans = {center: (start, stop) for (center, (start, stop)) in zip(centers, spans) if (start <= center < stop)}
        # insert '|' downward to each child
        text_cols = [list(col) for col in zip(*rows)]
        # extension_indices = set()  # indices at which to branch downward
        for (j, col) in enumerate(text_cols):
            if (span := column_spans.get(j)) is None:
                continue
            for (i, row) in enumerate(rows):
                if row[span[0]:span[1]].isspace():
                    col[i] = '│'
                else:
                    break
        rows = node_lines + [''.join(row) for row in zip(*text_cols)]
        top_line = node_lines[0]
        if top_line.isspace():
            new_center = center
        else:
            (leading, top_text, _) = _partition_line(node_lines[0])
            new_center = len(leading) + _center_index(len(top_text))
        return Column(width, new_center, rows)
    lines = tree.fold(conjoin_subtrees).rows
    # ensure the tree is padded on the left & right
    lines = _pad_lines(lines)
    return '\n'.join(lines)

# PLANAR DRAWING

class Box(NamedTuple):
    """Class representing a box (rectangle)."""
    x: float
    y: float
    width: float
    height: float

    def shift(self, dx: float, dy: float) -> Self:
        """Returns a new box, shifted by the given (dx, dy)."""
        return type(self)(self.x + dx, self.y + dy, self.width, self.height)

BoxPair = tuple[Box, Box]

@dataclass
class TreeLayout:
    """Class for laying out a tree diagram in 2D coordinates."""
    xchar: float = 0.16  # x width of characters
    ychar: float = 0.21  # y width of characters
    xgap: float = 1.0  # horizontal gap dimension
    ygap: float = 0.7  # vertical gap dimension
    ylead: float = 0.06  # space between lines of text
    ypad_top: float = 0.05  # space between node text and arrow above
    ypad_bottom: float = 0.09  # space between node text and arrow below (may exceed ypad_top because some characters descend below baseline)

    def text_size(self, s: str) -> tuple[float, float]:
        """Gets the width and height of a block of text."""
        lines = s.splitlines()
        num_lines = len(lines)
        if num_lines == 0:
            width = 0.0
        else:
            width = self.xchar * max(map(len, lines))
        height = self.ychar * num_lines + self.ylead * max(0, num_lines - 1)
        return (width, height)

    def _shift_coord_node_pair(self, dx: float, dy: float) -> Callable[[tuple[BoxPair, T]], tuple[BoxPair, T]]:
        def _shift(pair: tuple[BoxPair, T]) -> tuple[BoxPair, T]:
            ((box1, box2), node) = pair
            return ((box1.shift(dx, dy), box2.shift(dx, dy)), node)
        return _shift

    def tree_with_boxes(self, tree: BaseTree[T], *, top_down: bool = False) -> BaseTree[tuple[Box, T]]:
        """Computes bounding boxes for each node of the tree for the given drawing style.
        Returns a new tree of (box, node) pairs."""
        cls = cast(Type[BaseTree[tuple[BoxPair, T]]], type(tree))
        # recursively compute (parent node box, full subtree box) for each node
        def get_boxes(node: T, children: Sequence[BaseTree[tuple[BoxPair, T]]]) -> BaseTree[tuple[BoxPair, T]]:
            # compute parent dimensions from text size
            (parent_width, parent_height) = self.text_size(str(node))
            num_children = len(children)
            if num_children == 0:
                full_box = parent_box = Box(0.0, 0.0, parent_width, parent_height)
            else:
                child_widths = [child.node[0][1].width for child in children]
                # get x offsets for the child boxes, inserting gaps
                dxs = cumsums([self.xgap + child_width for child_width in child_widths] + [child_widths[-1]])
                children_width = dxs[-1]  # width of all children together
                dy = -(parent_height + self.ygap)
                child_heights = [child.node[0][1].height for child in children]
                max_child_height = max(child_heights)
                if top_down:  # same vertical offset for each child
                    dys = [dy] * num_children
                else:  # adjust vertical offset for each child's height
                    dys = [dy - (max_child_height - height) for height in child_heights]
                if parent_width > children_width:  # shift children under parent
                    full_width = parent_width
                    diff = (parent_width - children_width) / 2.0
                    # shift each child's box to the right by the appropriate offset
                    children = [child.map(self._shift_coord_node_pair(dx + diff, dy)) for (child, dx, dy) in zip(children, dxs, dys)]
                    parent_x = 0.0
                else:  # center parent over its children
                    full_width = children_width
                    children = [child.map(self._shift_coord_node_pair(dx, dy)) for (child, dx, dy) in zip(children, dxs, dys)]
                    (lbox, rbox) = (children[0].node[0][0], children[-1].node[0][0])
                    (left, right) = (lbox.x, rbox.x + rbox.width)
                    parent_x = (left + right - parent_width) / 2.0
                parent_box = Box(parent_x, 0.0, parent_width, parent_height)
                full_height = parent_height + self.ygap + max_child_height
                full_box = Box(0.0, 0.0, full_width, full_height)
            return cls(((parent_box, full_box), node), children)
        def discard_full_box(pair: tuple[BoxPair, T]) -> tuple[Box, T]:
            ((parent_box, _), node) = pair
            return (parent_box, node)
        return tree.fold(get_boxes).map(discard_full_box)
