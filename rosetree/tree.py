from __future__ import annotations

from abc import ABC
from collections import UserList, defaultdict
from collections.abc import Iterator, Sequence
from functools import cached_property, reduce
from itertools import chain
from operator import add, itemgetter
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Hashable, Literal, Optional, Type, TypedDict, TypeVar, Union, cast

from typing_extensions import NotRequired, Self

from .draw import BoxPair, TreeDrawOptions, TreeLayoutOptions, pretty_tree_long, pretty_tree_wide
from .trie import Trie
from .utils import merge_dicts


if TYPE_CHECKING:
    import networkx as nx


S = TypeVar('S')
T = TypeVar('T')
U = TypeVar('U')
H = TypeVar('H', bound=Hashable)

TreePath = tuple[T, ...]
TreeStyle = Literal[
    'top-down',
    'bottom-up',
    'long',
]

GraphData = tuple[int, dict[int, T], list[tuple[int, int]]]

class TreeDict(TypedDict):
    """Type representing the result of calling `to_dict` on a `BaseTree`."""
    n: Any
    c: NotRequired[list[TreeDict]]


class BaseTree(ABC, Sequence['BaseTree[T]']):
    """Base class for a simple tree, represented by a node and a sequence of child subtrees."""

    def __init__(self, node: T, children: Optional[Sequence[BaseTree[T]]] = None) -> None:
        self.node = node

    @classmethod
    def wrap(cls, obj: Union[T, BaseTree[T]], *, deep: bool = False) -> Self:
        """Wraps an object in a trivial singleton tree.
        If the object is already a BaseTree, returns the same tree (except possibly converting the top-level object to the target class).
        If deep=True, wraps each subtree recursively."""
        if isinstance(obj, BaseTree):
            if deep:
                return obj.fold(cls)
            else:
                if isinstance(obj, cls):
                    return obj
                # wrap the top layer only
                return cls(obj.node, list(obj))
        # non-tree object
        return cls(obj)

    @classmethod
    def unfold(cls, func: Callable[[S], tuple[T, Sequence[S]]], seed: S) -> BaseTree[T]:
        """Constructs a tree from an "unfolding" function and a seed.
        The function takes a seed as input and returns a (node, children) pair, where children is a list of new seed objects to be unfolded in the next round.
        The process terminates when every seed evaluates to have no children.
        This is also known as an *anamorphism* for the tree functor."""
        (node, children) = func(seed)
        return cls(node, [cls.unfold(func, child) for child in children])

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and (self.node == other.node)
            and (len(self) == len(other))
            and all(child == other_child for (child, other_child) in zip(self, other))
        )

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.node!r}, {list.__repr__(list(self))})'

    # FUNCTIONAL METHODS

    def map(self, f: Callable[[T], U]) -> BaseTree[U]:
        """Maps a function onto each node of a tree, preserving its structure."""
        cls = cast(Type[BaseTree[U]], type(self))
        children = [child.map(f) for child in self]
        return cls(f(self.node), children)

    def leaf_map(self, f: Callable[[T], U]) -> BaseTree[Union[T, U]]:
        """Maps a function onto each leaf node of the tree, preserving its structure.
        This results in a tree which may have mixed types."""
        cls = cast(Type[BaseTree[Union[T, U]]], type(self))
        children = [child.leaf_map(f) if isinstance(child, BaseTree) else f(child) for child in self]
        if not children:  # root is a leaf
            return cls(f(self.node), children)
        return cls(self.node, children)

    def internal_map(self, f: Callable[[T], U]) -> BaseTree[Union[T, U]]:
        """Maps a function onto each internal (non-leaf) node of the tree, preserving its structure.
        This results in a tree which may have mixed types."""
        cls = cast(Type[BaseTree[Union[T, U]]], type(self))
        children = [child.internal_map(f) if isinstance(child, BaseTree) else child for child in self]
        if children:  # root is an internal node
            return cls(f(self.node), children)
        return cls(self.node)

    def fold(self, f: Callable[[T, Sequence[U]], U]) -> U:
        """Folds a tree into a "summary" value.
        For each node in the tree, apply a binary function f to that node and the result of applying f to each of its child subtrees.
        This is also known as a *catamorphism* for the tree functor."""
        children = [child.fold(f) for child in self]
        return f(self.node, children)

    def reduce(self, f: Callable[[T, T], T], *, preorder: bool = True) -> T:
        """Given a binary operation, reduces the operation over all the nodes.
        If preorder=True, calls the function with the parent node as the first argument; otherwise calls it as the second argument."""
        children = [child.reduce(f, preorder=preorder) for child in self]
        if not children:
            return self.node
        reduced_children = reduce(f, children)
        return f(self.node, reduced_children) if preorder else f(reduced_children, self.node)

    def scan(self, f: Callable[[T, T], T], *, preorder: bool = True) -> BaseTree[T]:
        """Given a binary operation, creates a new tree where each node's value is the reduction of the operation over that node and all of its descendants.
        If preorder=True, calls the function with the parent node as the first argument; otherwise calls it as the second argument."""
        cls = type(self)
        def func(node: T, children: Sequence[BaseTree[T]]) -> BaseTree[T]:
            if not children:
                return cls(node)
            reduced_children = reduce(f, [child.node for child in children])
            # each child node has the reduced value, so reduce these together with the parent
            value = f(node, reduced_children) if preorder else f(reduced_children, node)
            return cls(value, children)
        return self.fold(func)

    def filter(self, pred: Callable[[T], bool]) -> Optional[Self]:
        """Generalized filter over a tree.
        Takes a predicate function on nodes returning a boolean (True if satisfied).
        Filters out all nodes which do not satisfy this predicate.
        If the root node does not satisfy the predicate, returns None."""
        if pred(self.node):
            children = [filtered for child in self if (filtered := child.filter(pred)) is not None]
            return type(self)(self.node, children)
        return None

    # PROPERTIES

    def is_leaf(self) -> bool:
        """Returns True if the root node has no children."""
        return len(self) == 0

    @property
    def leaves(self) -> list[T]:
        """Returns a list of the tree's leaves, in left-to-right order."""
        return list(self.iter_leaves())

    @property
    def height(self) -> int:
        """Returns the height (maximum distance from the root to any leaf) of the tree."""
        return self.tag_with_height().node[0]

    @property
    def size(self) -> int:
        """Returns the size (total number of nodes) of the tree."""
        return self.tag_with_size().node[0]

    def depth_sorted_nodes(self) -> Iterator[list[T]]:
        """Iterates through equivalence classes (lists) of nodes by increasing depth.
        Each successive list has nodes with depth one greater than that of the previous list."""
        yield [self.node]
        children = list(self)
        while children:
            yield [child.node for child in children]
            children = [grandchild for child in children for grandchild in child]

    def height_sorted_nodes(self) -> Iterator[list[T]]:
        """Iterates through equivalence classes (lists) of nodes by increasing height.
        Each successive list has nodes with height one greater than that of the previous list."""
        # NOTE: this is not easily done recursively, so we adopt a more brute-force strategy of enumerating all nodes with their height, and grouping them by height.
        nodes_by_height = defaultdict(list)
        for (height, node) in self.tag_with_height().iter_nodes():
            nodes_by_height[height].append(node)
        max_height = max(nodes_by_height)
        # all heights from 0..max_height should have at least one node
        for height in range(max_height + 1):
            yield nodes_by_height[height]

    # ITERATION

    def iter_nodes(self, *, preorder: bool = True) -> Iterator[T]:
        """Iterates over nodes.
        If preorder=True, does a pre-order traversal, otherwise post-order."""
        def _iter_nodes(node: T, children: Sequence[Iterator[T]]) -> Iterator[T]:
            child_iter = chain.from_iterable(children)
            return chain([node], child_iter) if preorder else chain(child_iter, [node])
        return self.fold(_iter_nodes)

    def iter_leaves(self) -> Iterator[T]:
        """Iterates over leaves, in left-to-right order."""
        def _iter_leaves(node: T, children: Sequence[Iterator[T]]) -> Iterator[T]:
            if len(children) == 0:
                return iter([node])
            return (leaf for child in children for leaf in child)
        return self.fold(_iter_leaves)

    def iter_edges(self, *, preorder: bool = True) -> Iterator[tuple[T, T]]:
        """Iterates over (parent, child) edges.
        If preorder=True, does a pre-order traversal, otherwise post-order."""
        def _iter_edges(node: T, children: Sequence[tuple[T, Iterator[tuple[T, T]]]]) -> tuple[T, Iterator[tuple[T, T]]]:
            if preorder:
                edges_iter = chain.from_iterable(chain([(node, child)], child_edges) for (child, child_edges) in children)
            else:
                edges_iter = chain.from_iterable(chain(child_edges, [(node, child)]) for (child, child_edges) in children)
            return (node, edges_iter)
        return self.fold(_iter_edges)[1]

    def iter_subtrees(self, *, preorder: bool = True) -> Iterator[BaseTree[T]]:
        """Iterates over subtrees of each node.
        If preorder=True, does a pre-order traversal, otherwise post-order."""
        child_iter = chain.from_iterable(child.iter_subtrees(preorder=preorder) for child in self)
        return chain([self], child_iter) if preorder else chain(child_iter, [self])

    def iter_paths(self, *, preorder: bool = True) -> Iterator[TreePath[T]]:
        """Iterates over paths from the root to each node.
        If preorder=True, does a pre-order traversal, otherwise post-order."""
        def _iter_paths(node: T, children: Sequence[Iterator[TreePath[T]]]) -> Iterator[TreePath[T]]:
            paths = [((node,) + path for path in tails) for tails in children]
            subtree_paths = chain.from_iterable(paths)
            return chain([(node,)], subtree_paths) if preorder else chain(subtree_paths, [(node,)])
        return self.fold(_iter_paths)

    def iter_full_paths(self) -> Iterator[TreePath[T]]:
        """Iterates over paths from the root to each leaf, in left-to-right order."""
        def _iter_full_paths(node: T, children: Sequence[Iterator[TreePath[T]]]) -> Iterator[TreePath[T]]:
            path_iters = children if (len(children) > 0) else [iter([()])]
            paths = [((node,) + path for path in tails) for tails in path_iters]
            return chain.from_iterable(paths)
        return self.fold(_iter_full_paths)

    # TRANSFORMATIONS

    def remove_leaves(self) -> Optional[Self]:
        """Removes all the leaves from the tree.
        This returns a new tree (does *not* update in-place).
        Nodes that were parents of leaves will now become leaves.
        Returns None if the root of the tree is itself a leaf."""
        if self.is_leaf():
            return None
        children = [subtree for child in self if (subtree := child.remove_leaves()) is not None]
        return type(self)(self.node, children)

    def prune_to_depth(self, max_depth: int) -> BaseTree[T]:
        """Prunes the tree to the given maximum depth (distance from root)."""
        if max_depth < 0:
            raise ValueError('max_depth must be a nonnegative integer')
        filtered = self.tag_with_depth().filter(lambda pair: pair[0] <= max_depth)
        assert filtered is not None  # (since depth of root is 0)
        return filtered.map(itemgetter(1))

    def tag_with_depth(self) -> BaseTree[tuple[int, T]]:
        """Converts each tree node to a pair (depth, node), where the depth of a node is the minimum distance to the root."""
        cls = cast(Type[BaseTree[tuple[int, T]]], type(self))
        def with_depth(node: T, children: Sequence[BaseTree[tuple[int, T]]]) -> BaseTree[tuple[int, T]]:
            tagged_children = [child.map(lambda pair: (pair[0] + 1, pair[1])) for child in children]
            return cls((0, node), tagged_children)
        return self.fold(with_depth)

    def tag_with_height(self) -> BaseTree[tuple[int, T]]:
        """Converts each tree node to a pair (height, node), where the height of a node is the maximum distance to any leaf."""
        cls = cast(Type[BaseTree[tuple[int, T]]], type(self))
        def with_height(node: T, children: Sequence[BaseTree[tuple[int, T]]]) -> BaseTree[tuple[int, T]]:
            if len(children) == 0:
                height = 0
            else:
                height = max(child.node[0] for child in children) + 1
            return cls((height, node), children)
        return self.fold(with_height)

    def tag_with_size(self) -> BaseTree[tuple[int, T]]:
        """Converts each tree node to a pair (size, node), where the size of a node is the total number of nodes in that node's subtree (including the node itself)."""
        cls = cast(Type[BaseTree[tuple[int, T]]], type(self))
        def with_size(node: T, children: Sequence[BaseTree[tuple[int, T]]]) -> BaseTree[tuple[int, T]]:
            size = sum(child.node[0] for child in children) + 1
            return cls((size, node), children)
        return self.fold(with_size)

    def tag_with_unique_counter(self, *, preorder: bool = True) -> BaseTree[tuple[int, T]]:
        """Converts each tree node to a pair (id, node), where id is an incrementing integer uniquely identifying each node.
        If preorder=True, traverses in pre-order fashion, otherwise post-order."""
        cls = cast(Type[BaseTree[tuple[int, T]]], type(self))
        ctr = 0
        def tag_with_ctr(node: T) -> tuple[int, T]:
            nonlocal ctr
            pair = (ctr, node)
            ctr += 1
            return pair
        def map_tag(tree: BaseTree[T]) -> BaseTree[tuple[int, T]]:
            if preorder:
                parent = tag_with_ctr(tree.node)
                children = [map_tag(child) for child in tree]
            else:
                children = [map_tag(child) for child in tree]
                parent = tag_with_ctr(tree.node)
            return cls(parent, children)
        return map_tag(self)

    def to_path_tree(self) -> BaseTree[tuple[T, ...]]:
        """Converts each tree node to a path from the root to that node.
        Each path is represented as a tuple of nodes starting from the root."""
        paths = list(self.iter_paths(preorder=True))
        def get_path(pair: tuple[int, T]) -> tuple[T, ...]:
            return paths[pair[0]]
        return self.tag_with_unique_counter().map(get_path)

    # DRAWING

    def pretty(self, style: TreeStyle = 'top-down') -> str:
        """Generates a "pretty" ASCII representation of the tree.
        The following styles are supported:
            - `top-down`: (default) positions nodes of the same depth on the same vertical level
            - `bottom-up`: positions leaf nodes on the same vertical level
            - `long`: each node is shown on its own line (this is analogous to the Linux `tree` command for viewing files and directories)"""
        if style == 'top-down':
            return pretty_tree_wide(self, top_down=True)
        if style == 'bottom-up':
            return pretty_tree_wide(self, top_down=False)
        if style == 'long':
            return pretty_tree_long(self)
        raise ValueError(f'invalid pretty tree style {style!r}')

    def with_bounding_boxes(self, style: TreeStyle = 'top-down', *, layout_options: Optional[TreeLayoutOptions] = None) -> BaseTree[tuple[BoxPair, T]]:
        """Computes bounding boxes for each node of the tree for the given drawing style.
        You may optionally provide a TreeLayoutOptions object specifying various spacing options for laying out the drawing.
        Returns a new tree of ((parent box, full box), node) pairs."""
        if layout_options is None:  # use default layout
            layout_options = TreeLayoutOptions()
        if style == 'top-down':
            return layout_options.tree_with_boxes(self, top_down=True)
        if style == 'bottom-up':
            return layout_options.tree_with_boxes(self, top_down=False)
        raise ValueError(f'invalid style {style!r}')

    def draw(
        self,
        filename: Optional[str] = None,
        *,
        style: TreeStyle = 'top-down',
        draw_options: Optional[TreeDrawOptions] = None,
    ) -> None:
        """Draws a plot of the tree with matplotlib.
        If a filename is provided, saves it to this file; otherwise, displays the plot.
        If style='top-down', positions nodes of the same depth on the same vertical level.
        If style='bottom-up', positions leaf nodes on the same vertical level."""
        if draw_options is None:  # use default options
            draw_options = TreeDrawOptions()
        tree_with_boxes = self.with_bounding_boxes(style=style, layout_options=draw_options.layout_options)
        draw_options.draw(tree_with_boxes, filename=filename)

    # CONVERSION

    def to_dict(self) -> TreeDict:
        """Converts the tree to a Python dict.
        The dict contains two fields:
            - `"n"`, with the node object,
            - `"c"`, with a list of dicts representing the child subtrees.
        Leaf nodes will omit the `"c"` entry.
        This is useful for things like JSON serialization."""
        def _to_dict(node: T, children: Sequence[TreeDict]) -> TreeDict:
            d: TreeDict = {'n': node}
            if len(children) > 0:
                d['c'] = list(children)
            return d
        return self.fold(_to_dict)

    @classmethod
    def from_dict(cls, d: TreeDict) -> Self:
        """Constructs a tree from a Python dict.
        See `BaseTree.to_dict` for more details on the structure."""
        return cls(d['n'], [cls.from_dict(child) for child in d.get('c', [])])

    def to_networkx(self) -> nx.DiGraph[int]:
        """Converts the tree to a networkx.DiGraph.
        The nodes will be labeled with sequential integer IDs, and each node will have a 'data' field containing the original node data."""
        import networkx as nx
        def get_graph_data(node: tuple[int, T], children: Sequence[GraphData[T]]) -> GraphData[T]:
            node_id = node[0]
            all_nodes = {node_id: node[1]}
            if len(children) == 0:
                all_edges = []
            else:
                (child_ids, nodes, edges) = zip(*children)
                all_nodes.update(reduce(merge_dicts, nodes))  # type: ignore[arg-type]
                all_edges = [(node_id, child_id) for child_id in child_ids] + reduce(add, edges)
            return (node_id, all_nodes, all_edges)
        (_, nodes, edges) = self.tag_with_unique_counter().fold(get_graph_data)
        dg: nx.DiGraph[int] = nx.DiGraph()
        for (node_id, node) in nodes.items():
            dg.add_node(node_id, data=node)
        dg.add_edges_from(edges)
        return dg

    @classmethod
    def from_trie(cls, trie: Trie[T]) -> BaseTree[tuple[bool, tuple[T, ...]]]:
        """Constructs a tree from a Trie (prefix tree object).
        Nodes are (member, prefix) pairs, where member is a boolean indicating whether the prefix is in the trie."""
        node = (trie.member, ())
        pairs = [(sym, cls.from_trie(subtrie)) for (sym, subtrie) in trie.children.items()]
        def prepend_sym(sym: T) -> Callable[[tuple[bool, tuple[T, ...]]], tuple[bool, tuple[T, ...]]]:
            def prepend(pair: tuple[bool, tuple[T, ...]]) -> tuple[bool, tuple[T, ...]]:
                (member, tup) = pair
                return (member, (sym,) + tup)
            return prepend
        return cls(node, [child.map(prepend_sym(sym)) for (sym, child) in pairs])  # type: ignore


class Tree(BaseTree[T], UserList[T]):
    """A simple tree class, represented by a node and a list of child subtrees."""

    def __init__(self, node: T, children: Optional[Sequence[BaseTree[T]]] = None) -> None:
        """Creates a new tree from a node and child subtrees."""
        UserList.__init__(self, children or [])  # type: ignore[misc]
        BaseTree.__init__(self, node, children)


class FrozenTree(BaseTree[H], tuple[H, tuple['FrozenTree[H]', ...]]):
    """An immutable, hashable tree class, represented by a tuple (node, children).
    Each node must be a hashable object; children is a tuple of child subtrees."""

    def __new__(cls, node: H, children: Optional[Sequence[FrozenTree[H]]] = None) -> Self:  # noqa: D102
        return tuple.__new__(cls, (node, tuple(children) if children else ()))

    def __init__(self, node: H, children: Optional[Sequence[FrozenTree[H]]] = None) -> None:
        pass

    @property
    def node(self) -> H:  # type: ignore[override]  # noqa: D102
        return tuple.__getitem__(self, 0)  # type: ignore[return-value]

    def __len__(self) -> int:
        return len(tuple.__getitem__(self, 1))  # type: ignore[arg-type]

    def __getitem__(self, idx: Union[int, slice]) -> Union[FrozenTree[H], tuple[FrozenTree[H], ...]]:  # type: ignore[override]
        return tuple.__getitem__(self, 1).__getitem__(idx)  # type: ignore

    def __iter__(self) -> Iterator[FrozenTree[H]]:  # type: ignore[override]
        yield from tuple.__getitem__(self, 1)  # type: ignore[misc]

    @cached_property
    def _hash(self) -> int:
        return tuple.__hash__(self)

    def __hash__(self) -> int:
        return self._hash

    def tag_with_hash(self) -> FrozenTree[tuple[int, H]]:
        """Converts each tree node to a pair (hash, node), where hash is a hash that depends on the node's entire subtree."""
        cls = cast(Type[FrozenTree[tuple[int, H]]], type(self))
        def with_hash(node: H, children: Sequence[FrozenTree[tuple[int, H]]]) -> FrozenTree[tuple[int, H]]:
            h = hash((node, tuple(children)))
            return cls((h, node), children)
        return self.fold(with_hash)


class MemoTree(FrozenTree[H]):
    """An immutable, hashable tree class which memoizes all unique instances.
    This can conserve memory in the case where a large number of identical trees is created."""

    _instances: ClassVar[dict[Any, Any]] = {}

    def __new__(cls, node: H, children: Optional[Sequence[FrozenTree[H]]] = None) -> Self:  # noqa: D102
        children = tuple(children) if children else ()
        key = (node, children)
        try:
            return cls._instances[key]  # type: ignore[no-any-return]
        except KeyError:
            return cls._instances.setdefault(key, tuple.__new__(cls, key))  # type: ignore[no-any-return]


# FUNCTIONAL CONSTRUCTS

def zip_trees_with(f: Callable[..., U], *trees: BaseTree[T]) -> BaseTree[U]:
    """Given an n-ary function and n trees of the same shape, returns a new tree which applies the function to corresponding nodes of the tree."""
    if not trees:
        raise ValueError('must provide one or more trees')
    cls = cast(Type[BaseTree[U]], type(trees[0]))
    if len({len(tree) for tree in trees}) > 1:
        raise ValueError('trees must all have the same shape')
    node = f(*(tree.node for tree in trees))
    # iterate through tuple of subtrees for each position and recursively call zip_trees_with on each one
    children = [zip_trees_with(f, *subtrees) for subtrees in zip(*trees)]
    return cls(node, children)

def zip_trees(*trees: BaseTree[T]) -> BaseTree[tuple[T, ...]]:
    """Given one or more trees of the same shape, returns a tree of tuples of corresponding nodes."""
    f = lambda *args: args
    return zip_trees_with(f, *trees)
