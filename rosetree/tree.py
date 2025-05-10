from __future__ import annotations

from abc import ABC
from collections import UserList
from collections.abc import Iterator, Sequence
from functools import cached_property, reduce
from itertools import accumulate, chain
from operator import add, itemgetter
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Hashable, Optional, Type, TypedDict, TypeVar, Union, cast

from typing_extensions import NotRequired, Self

from .trie import Trie
from .utils import merge_dicts


if TYPE_CHECKING:
    import networkx as nx


S = TypeVar('S')
T = TypeVar('T')
U = TypeVar('U')
H = TypeVar('H', bound=Hashable)

# color string or RGB(A) tuple
ColorType = Union[str, tuple[float, ...]]
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

    def reduce(self, f: Callable[[T, T], T]) -> T:
        """Given an associative binary operation, reduces the operation over all the nodes."""
        children = [child.reduce(f) for child in self]
        return reduce(f, [self.node] + children)

    def aggregate_reduce(self, f: Callable[[T, T], T]) -> BaseTree[T]:
        """Given an associative binary operation, creates a new tree where each node's value is the reduction of the operation over that node and all of its descendants."""
        cls = type(self)
        def func(node: T, children: Sequence[BaseTree[T]]) -> BaseTree[T]:
            # each child node has the reduced value, so reduce these together with the parent
            value = reduce(f, [node] + [child.node for child in children])
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
        def func(node: T, children: Sequence[list[T]]) -> list[T]:
            if len(children) == 0:
                return [node]
            return [leaf for child in children for leaf in child]
        return self.fold(func)

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

    # ITERATION

    def iter_nodes(self, *, preorder: bool = True) -> Iterator[T]:
        """Iterates over nodes.
        If preorder=True, does a pre-order traversal, otherwise post-order."""
        def _iter_nodes(node: T, children: Sequence[Iterator[T]]) -> Iterator[T]:
            child_iter = chain.from_iterable(children)
            return chain([node], child_iter) if preorder else chain(child_iter, [node])
        return self.fold(_iter_nodes)

    def iter_subtrees(self, *, preorder: bool = True) -> Iterator[BaseTree[T]]:
        """Iterates over subtrees of each node.
        If preorder=True, does a pre-order traversal, otherwise post-order."""
        child_iter = chain.from_iterable(child.iter_subtrees(preorder=preorder) for child in self)
        return chain([self], child_iter) if preorder else chain(child_iter, [self])

    # TRANSFORMATIONS

    def defoliate(self) -> Optional[Self]:
        """Removes all the leaves from the tree.
        Nodes that were parents of leaves will now become leaves.
        Returns None if the root of the tree is itself a leaf."""
        if self.is_leaf():
            return None
        children = [subtree for child in self if (subtree := child.defoliate()) is not None]
        return type(self)(self.node, children)

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
        def incr(i: int) -> Callable[[tuple[int, tuple[int, T]]], tuple[int, tuple[int, T]]]:
            return lambda pair: (pair[0] + i, pair[1])
        def with_ctr(node: tuple[int, T], children: Sequence[BaseTree[tuple[int, tuple[int, T]]]]) -> BaseTree[tuple[int, tuple[int, T]]]:
            # get sizes of child subtrees
            sizes = [child.node[1][0] for child in children]
            # get cumulative sums of child subtree sizes
            cumsizes = list(accumulate([0] + sizes, add))
            cls = cast(Type[BaseTree[tuple[int, tuple[int, T]]]], type(self))
            if preorder:  # parent ID comes before descendants'
                ctr = 0
                new_children = [child.map(incr(i + 1)) for (i, child) in zip(cumsizes, children)]
            else:  # parent ID comes after descendants'
                ctr = cumsizes[-1]
                new_children = [child.map(incr(i)) for (i, child) in zip(cumsizes, children)]
            return cls((ctr, node), new_children)
        return self.tag_with_size().fold(with_ctr).map(lambda pair: (pair[0], pair[1][1]))

    def prune_to_depth(self, max_depth: int) -> BaseTree[T]:
        """Prunes the tree to the given maximum depth (distance from root)."""
        if max_depth < 0:
            raise ValueError('max_depth must be a nonnegative integer')
        filtered = self.tag_with_depth().filter(lambda pair: pair[0] <= max_depth)
        assert filtered is not None  # (since depth of root is 0)
        return filtered.map(itemgetter(1))

    # DRAWING

    def pretty(self, *, top_down: bool = False) -> str:
        """Generates a "pretty" ASCII representation of the tree.
        If top_down=True, positions nodes of the same depth on the same vertical level.
        Otherwise, positions leaf nodes on the same vertical level."""
        raise NotImplementedError

    def draw(
        self,
        filename: Optional[str] = None,
        *,
        top_down: bool = False,
        color: ColorType,
        leaf_color: Optional[ColorType] = None,
        edge_color: ColorType = 'black',
        linewidth: float = 1.0,
    ) -> None:
        """Draws a plot of the tree with matplotlib.
        If a filename is provided, saves it to this file; otherwise, displays the plot.
        If top_down=True, positions nodes of the same depth on the same vertical level.
        Otherwise, positions leaf nodes on the same vertical level.
        color: color (string or RGB tuple) of the node text
        leaf_color: color of the leaf node text (if None, same as color)
        edge_color: color of the edges
        linewidth: thickness of the edges"""
        raise NotImplementedError

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
