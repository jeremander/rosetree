from dataclasses import dataclass
import json
from operator import add, itemgetter, mul, sub
from types import GeneratorType

import pytest

from rosetree import FrozenTree, MemoTree, Tree, zip_trees, zip_trees_with


# BaseTree subclasses to test
TREE_CLASSES = [Tree, FrozenTree, MemoTree]

def tree_example1(cls):
    return cls(0, [cls(1), cls(2, [cls(3, [cls(4, [cls(5)])]), cls(6, [cls(7), cls(8)])])])

TREE1 = tree_example1(Tree)


def test_wrap():
    """Tests the wrap constructor."""
    tree1 = TREE1
    assert Tree.wrap(tree1) is tree1
    tree2 = FrozenTree.wrap(tree1)
    assert type(tree2) is FrozenTree
    assert tree2 != tree1
    assert type(tree2[0]) is Tree  # wrap only changes type of top-level tree
    tree3 = Tree.wrap(tree2)
    assert tree3 == tree1
    assert tree3 is not tree1
    assert Tree.wrap(1) == Tree(1)
    # deep wrap changes the type of all the subtrees
    tree4 = Tree.wrap(tree1, deep=True)
    assert tree4 == tree1
    assert tree4 is not tree1
    assert tree4[0] is not tree1[0]
    tree5 = FrozenTree.wrap(tree1, deep=True)
    assert type(tree5) is FrozenTree
    assert tree5 != tree1
    assert type(tree5[0]) is FrozenTree

def test_unfold():
    """Tests the unfold constructor."""
    def func(n):
        if n <= 1:
            return (n, [])
        half_n = n // 2
        return (n, [half_n, n - half_n])
    tree = Tree.unfold(func, 7)
    assert tree == Tree(7, [Tree(3, [Tree(1), Tree(2, [Tree(1), Tree(1)])]), Tree(4, [Tree(2, [Tree(1), Tree(1)]), Tree(2, [Tree(1), Tree(1)])])])

@pytest.mark.parametrize('cls', TREE_CLASSES)
def test_properties(cls):
    """Tests various properties of an example tree."""
    tree = tree_example1(cls)
    assert tree.node == 0
    assert tree.size == 9
    assert len(tree) == 2
    assert len(tree[0]) == 0
    assert tree[0].is_leaf()
    assert tree[1].node == 2
    assert tree.height == 4
    assert list(tree.depth_sorted_nodes()) == [[0], [1, 2], [3, 6], [4, 7, 8], [5]]
    assert list(tree.height_sorted_nodes()) == [[1, 5, 7, 8], [4, 6], [3], [2], [0]]
    assert type(tree.iter_leaves()) is GeneratorType
    assert tree.leaves == [1, 5, 7, 8]
    assert tree.leaves == list(tree.iter_leaves())
    assert list(tree.iter_nodes()) == list(range(9))
    assert list(tree.iter_nodes(preorder=False)) == [1, 5, 4, 3, 7, 8, 6, 2, 0]
    assert list(tree.iter_edges()) == [(0, 1), (0, 2), (2, 3), (3, 4), (4, 5), (2, 6), (6, 7), (6, 8)]
    assert list(tree.iter_edges(preorder=False)) == [(0, 1), (4, 5), (3, 4), (2, 3), (6, 7), (6, 8), (2, 6), (0, 2)]
    assert [subtree.node for subtree in tree.iter_subtrees()] == list(range(9))

@pytest.mark.parametrize('cls', TREE_CLASSES)
def test_hash_and_eq(cls):
    """Tests hashing and equality of trees."""
    tree1 = tree_example1(cls)
    tree2 = tree_example1(cls)
    assert tree1 == tree2
    if cls in [FrozenTree, MemoTree]:
        assert hash(tree1) == hash(tree2)
    else:
        with pytest.raises(TypeError, match='unhashable type'):
            _ = hash(tree1)
    if cls is MemoTree:
        assert tree1 is tree2
    else:
        assert tree1 is not tree2
    # converting int to str makes the trees unequal
    tree3 = tree1.map(lambda x: str(x) if (x == 0) else x)
    assert tree3 != tree1
    class MyInt(int):
        pass
    # wrapping int into custom subclass keeps the trees equal
    tree4 = tree1.map(MyInt)
    assert tree4 == tree1
    if cls is MemoTree:
        assert tree4 is tree1
    else:
        assert tree4 is not tree1
    # custom __eq__ to equate int with custom class
    @dataclass(frozen=True)
    class MyObj:
        i: int
        def __eq__(self, other):
            if isinstance(other, int):
                return self.i == other
            return super().__eq__(other)
    tree5 = tree1.map(MyObj)
    assert tree5 == tree1
    assert tree5 is not tree1

@pytest.mark.parametrize('cls', TREE_CLASSES)
def test_repr_eval(cls):
    """Tests that repr and eval are inverses."""
    tree = tree_example1(cls)
    tree_str = repr(tree)
    assert eval(tree_str) == tree
    assert repr(eval(tree_str)) == tree_str

@pytest.mark.parametrize('cls', TREE_CLASSES)
def test_leaf_and_internal_map(cls):
    """Tests the leaf_map and internal_map methods."""
    tree1 = tree_example1(cls)
    tree2 = tree1.leaf_map(str)
    assert tree2.leaves == [str(i) for i in tree1.leaves]
    tree3 = tree1.internal_map(str)
    assert tree3.leaves == tree1.leaves
    assert all(isinstance(node, str) for node in tree3.remove_leaves().iter_nodes())

@pytest.mark.parametrize('cls', TREE_CLASSES)
def test_reduce(cls):
    """Tests the reduce method."""
    tree = tree_example1(cls)
    assert tree.reduce(add, preorder=True) == 36
    assert tree.reduce(add, preorder=False) == 36
    assert tree[1].reduce(mul, preorder=True) == 40320
    assert tree[1].reduce(mul, preorder=False) == 40320
    assert tree.reduce(sub, preorder=True) == 4
    assert tree.reduce(sub, preorder=False) == -2

def test_scan():
    """Tests the scan method (e.g. creating a "sum tree" of subtree sums)."""
    tree1 = TREE1
    sum_tree1 = tree1.scan(add)
    assert sum_tree1 == Tree(36, [Tree(1), Tree(35, [Tree(12, [Tree(9, [Tree(5)])]), Tree(21, [Tree(7), Tree(8)])])])
    assert sum_tree1.node == tree1.reduce(add)
    # more common use case is when only the leaves are nonzero
    tree2 = TREE1.internal_map(lambda _: 0)
    sum_tree2 = tree2.scan(add)
    assert sum_tree2 == Tree(21, [Tree(1), Tree(20, [Tree(5, [Tree(5, [Tree(5)])]), Tree(15, [Tree(7), Tree(8)])])])
    assert sum_tree2.node == tree2.reduce(add)
    # also test a non-commutative operation (subtraction)
    diff_tree1 = tree1.scan(sub)
    assert diff_tree1 == Tree(4, [Tree(1), Tree(5, [Tree(4, [Tree(-1, [Tree(5)])]), Tree(7, [Tree(7), Tree(8)])])])
    assert diff_tree1.node == tree1.reduce(sub)
    diff_tree2 = tree1.scan(sub, preorder=False)
    assert diff_tree2 == Tree(-2, [Tree(1), Tree(3, [Tree(-2, [Tree(1, [Tree(5)])]), Tree(-7, [Tree(7), Tree(8)])])])
    assert diff_tree2.node == tree1.reduce(sub, preorder=False)

@pytest.mark.parametrize('cls', TREE_CLASSES)
def test_filter(cls):
    """Tests the filter method."""
    tree = tree_example1(cls)
    assert tree.filter(lambda x: x % 2 == 0) == cls(0, [cls(2, [cls(6, [cls(8)])])])
    assert tree.filter(lambda x: x % 2 == 1) is None
    assert tree.filter(lambda x: x <= 4) == cls(0, [cls(1), cls(2, [cls(3, [cls(4)])])])

def test_iter_paths():
    """Tests the iter_paths and iter_full_paths methods."""
    tree = TREE1
    paths_pre = list(tree.iter_paths(preorder=True))
    assert paths_pre == [
        (0,),
        (0, 1),
        (0, 2),
        (0, 2, 3),
        (0, 2, 3, 4),
        (0, 2, 3, 4, 5),
        (0, 2, 6),
        (0, 2, 6, 7),
        (0, 2, 6, 8)
    ]
    assert [path[-1] for path in paths_pre] == list(tree.iter_nodes(preorder=True))
    paths_post = list(tree.iter_paths(preorder=False))
    assert paths_post == [
        (0, 1),
        (0, 2, 3, 4, 5),
        (0, 2, 3, 4),
        (0, 2, 3),
        (0, 2, 6, 7),
        (0, 2, 6, 8),
        (0, 2, 6),
        (0, 2),
        (0,)
    ]
    assert [path[-1] for path in paths_post] == list(tree.iter_nodes(preorder=False))
    assert set(paths_pre) == set(paths_post)
    paths_full = list(tree.iter_full_paths())
    assert paths_full == [
        (0, 1),
        (0, 2, 3, 4, 5),
        (0, 2, 6, 7),
        (0, 2, 6, 8)
    ]
    assert [path[-1] for path in paths_full] == tree.leaves

@pytest.mark.parametrize('cls', TREE_CLASSES)
def test_remove_leaves(cls):
    """Tests the remove_leaves method."""
    tree1 = tree_example1(cls)
    tree2 = tree1.remove_leaves()
    assert tree2 == cls(0, [cls(2, [cls(3, [cls(4)]), cls(6)])])

@pytest.mark.parametrize('cls', TREE_CLASSES)
def test_prune_to_depth(cls):
    """Tests the prune_to_depth method."""
    tree = tree_example1(cls)
    with pytest.raises(ValueError, match='max_depth must be a nonnegative integer'):
        _ = tree.prune_to_depth(-1)
    assert tree.prune_to_depth(0) == cls(0)
    assert tree.prune_to_depth(1) == cls(0, [cls(1), cls(2)])
    assert tree.prune_to_depth(2) == cls(0, [cls(1), cls(2, [cls(3), cls(6)])])
    tree4 = tree.prune_to_depth(4)
    assert tree4 == tree
    tree5 = tree.prune_to_depth(5)
    assert tree5 == tree
    if cls is MemoTree:
        assert tree4 is tree
        assert tree5 is tree
    else:
        assert tree4 is not tree
        assert tree5 is not tree

def test_tag_with_unique_counter():
    """Tests the tag_with_unique_counter method."""
    tree2 = TREE1.tag_with_unique_counter(preorder=True)
    tags = list(tree2.map(itemgetter(0)).iter_nodes())
    assert tags == list(range(TREE1.size))
    tree2 = TREE1.tag_with_unique_counter(preorder=False)
    tags = list(tree2.map(itemgetter(0)).iter_nodes())
    assert set(tags) == set(range(TREE1.size))
    assert tags == [8, 0, 7, 3, 2, 1, 6, 4, 5]

@pytest.mark.parametrize('cls', TREE_CLASSES)
def test_tag_with_hash(cls):
    """Tests the tag_with_hash method (for FrozenTree)."""
    if issubclass(cls, FrozenTree):
        tree = tree_example1(cls)
        tree2 = tree.tag_with_hash()
        leaves = set(tree.leaves)
        for (h, node) in tree2.iter_nodes():
            if node in leaves:
                # check leaf hash matches what we expect
                assert h == hash((node, ()))
    else:
        assert not hasattr(cls, 'tag_with_hash')

def test_to_path_tree():
    """Tests the to_path_tree method."""
    tree2 = TREE1.to_path_tree()
    assert tree2 == Tree((0,), [
        Tree((0, 1)),
        Tree((0, 2), [
            Tree((0, 2, 3), [
                Tree((0, 2, 3, 4), [
                    Tree((0, 2, 3, 4, 5))
                ])
            ]),
            Tree((0, 2, 6), [
                Tree((0, 2, 6, 7)),
                Tree((0, 2, 6, 8)),
            ])
        ])
    ])
    assert tree2.map(lambda path: path[-1]) == TREE1
    for preorder in [False, True]:
        assert list(TREE1.iter_paths(preorder=preorder)) == list(tree2.iter_nodes(preorder=preorder))

def test_tag_with_index_path():
    """Tests the tag_with_index_path method."""
    tree2 = TREE1.tag_with_index_path()
    assert tree2 == Tree(((0,), 0), [
        Tree(((0, 0), 1)),
        Tree(((0, 1), 2), [
            Tree(((0, 1, 0), 3), [
                Tree(((0, 1, 0, 0), 4), [
                    Tree(((0, 1, 0, 0, 0), 5))
                ])
            ]),
            Tree(((0, 1, 1), 6), [
                Tree(((0, 1, 1, 0), 7)),
                Tree(((0, 1, 1, 1), 8)),
            ])
        ])
    ])
    assert tree2.map(itemgetter(1)) == TREE1
    # index paths in preorder traversal order are sorted
    idx_paths = [idx_path for (idx_path, _) in tree2.iter_nodes()]
    assert sorted(idx_paths) == idx_paths

@pytest.mark.parametrize('cls', TREE_CLASSES)
def test_dict(cls):
    """Tests conversion to/from a dict."""
    tree1 = tree_example1(cls)
    d = tree1.to_dict()
    assert json.dumps(d) == '{"n": 0, "c": [{"n": 1}, {"n": 2, "c": [{"n": 3, "c": [{"n": 4, "c": [{"n": 5}]}]}, {"n": 6, "c": [{"n": 7}, {"n": 8}]}]}]}'
    assert isinstance(d, dict)
    tree2 = cls.from_dict(d)
    assert tree2 == tree1
    assert tree2.to_dict() == d

def test_networkx():
    """Tests the to_networkx method (converting to a networkx.DiGraph)."""
    dg = TREE1.to_networkx()
    assert list(dg.nodes) == list(range(TREE1.size))
    assert list(dg.edges) == [(0, 1), (0, 2), (2, 3), (2, 6), (3, 4), (4, 5), (6, 7), (6, 8)]

def test_zip_trees():
    """Tests the zip_trees and zip_trees_with functions."""
    square = lambda x: x ** 2
    tree0 = TREE1.map(lambda _: 1)
    tree1 = TREE1
    tree2 = TREE1.map(square)
    # 0 arguments
    with pytest.raises(ValueError, match='must provide one or more trees'):
        _ = zip_trees_with(lambda: 1)
    with pytest.raises(ValueError, match='must provide one or more trees'):
        _ = zip_trees()
    # 1 argument
    assert zip_trees_with(square, tree1) == tree2
    assert zip_trees(tree1) == tree1.map(lambda x: (x,))
    # 2 arguments
    assert zip_trees_with(add, tree1, tree2) == tree1.map(lambda x: x + x ** 2)
    assert zip_trees(tree1, tree1) == tree1.map(lambda x: (x, x))
    # 3 arguments
    add3 = lambda x1, x2, x3: x1 + x2 + x3
    assert zip_trees_with(add3, tree0, tree1, tree2) == tree1.map(lambda x: 1 + x + x ** 2)
    assert zip_trees(tree1, tree1, tree1) == tree1.map(lambda x: (x, x, x))
    # mismatched shape
    with pytest.raises(ValueError, match='trees must all have the same shape'):
        _ = zip_trees_with(add, tree1, tree1[0])
    with pytest.raises(ValueError, match='trees must all have the same shape'):
        _ = zip_trees(tree1, tree1.remove_leaves())
