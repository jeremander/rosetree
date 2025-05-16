# rosetree

[![PyPI - Version](https://img.shields.io/pypi/v/rosetree)](https://pypi.org/project/rosetree/)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/jeremander/rosetree/workflow.yml)
![Coverage Status](https://github.com/jeremander/rosetree/raw/coverage-badge/coverage-badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://raw.githubusercontent.com/jeremander/rosetree/refs/heads/main/LICENSE)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

## Installation

`pip install rosetree`

Or, to ensure you have all the required dependencies for tree drawing:

`pip install rosetree[draw]`

## Basics

`rosetree` provides a generic *multi-way tree* ("rose tree") data structure. It is inspired by Haskell's [Data.Tree](https://hackage-content.haskell.org/package/containers-0.8/docs/Data-Tree.html) library and is well suited to programming in a *functional* style.

We'll proceed by way of examples. First, let's create a simple `Tree` with integer node labels:

```python
from rosetree import Tree

# build the tree
>>> tree = Tree(1, [Tree(2, [Tree(3), Tree(4)]), Tree(5)])

# draw the tree
>>> print(tree.pretty())
    1
  ┌─┴──┐
  2    5
 ┌┴─┐
 3  4
```

We construct a `Tree` by specifying its root node and a list of child subtrees. A leaf node is therefore represented simply by a `Tree` with a node and no children.

### Drawing

The `pretty` method of a `Tree` produces an "ASCII art" string that can be used to view the tree's structure. There are three styles you can choose from by providing a `style` argument to `pretty`:

```python
# (this is the default style)
>>> print(tree.pretty(style='top-down'))
    1
  ┌─┴──┐
  2    5
 ┌┴─┐
 3  4

>>> print(tree.pretty(style='bottom-up'))
    1
  ┌─┴──┐
  2    │
 ┌┴─┐  │
 3  4  5

>>> print(tree.pretty(style='long'))
1
├── 2
│   ├── 3
│   └── 4
└── 5
```

Alternatively you can create an image representation of the tree (this requires `matplotlib`):

```python
# pop up a GUI window
>>> tree.draw()

# or just save a file directly
>>> tree.draw('my_tree.png')
```

<img src="doc/tree.png" width="180" alt="Picture of a Tree produced by Tree.draw"/>

You may also provide a `draw_options` argument to customize some aspects of the tree drawing such as node color, edge color, text font, spacing, etc.

### Accessing elements and properties

A `Tree` is actually implemented as a simple Python list of child subtrees, plus an additional `node` attribute to access the top-level node:

```python
# this is the root node
>>> tree.node
1

# this is the left-hand child subtree
>>> tree[0]
Tree(2, [Tree(3, []), Tree(4, [])])

# this is the right subtree of the left subtree
>>> tree[0][1]
Tree(4, [])

# this is the node at that subtree
>>> tree[0][1].node
4

# this is the number of child subtrees of the root node
>>> len(tree)
2
```

A few other `Tree` methods can be used to calculate properties of the tree:

```python
# total number of nodes in the tree
# NOTE: this is *not* the same as len(tree)!
>>> tree.size
5

# maximum distance from the root to any leaf
>>> tree.height
2
```

You can iterate over nodes, leaves, or subtrees:

```python
# iterate nodes with "pre-order" traversal (parents before children)
>>> list(tree.iter_nodes())
[1, 2, 3, 4, 5]

# iterate nodes with "post-order" traversal (children before parents)
>>> list(tree.iter_nodes(preorder=False))
[3, 4, 2, 5, 1]

# iterate leaves in left-to-right order
>>> list(tree.iter_leaves())
[3, 4, 5]

# iterate all subtrees with pre-order traversal
>>> list(tree.iter_subtrees())
[Tree(1, [Tree(2, [Tree(3, []), Tree(4, [])]), Tree(5, [])]),
 Tree(2, [Tree(3, []), Tree(4, [])]),
 Tree(3, []),
 Tree(4, []),
 Tree(5, [])]
```

**Note**: all of these `iter_` methods produce a *lazy* generator to avoid storing all the items in memory. You can step through the generator with a for loop, or generate all the elements by calling `list`, as in the examples above.

For convenience, you can call `tree.leaves` to get a list of all the leaf nodes.

### Inserting and deleting elements

A `Tree`, being a Python `list`, is capable of inserting or deleting elements fairly easily.

```python
from copy import deepcopy

# make a copy of the tree, since we'll be modifying it
>>> tree_copy = deepcopy(tree)
>>> print(tree_copy.pretty())
    1
  ┌─┴──┐
  2    5
 ┌┴─┐
 3  4

# insert another child node below the root
>>> tree_copy.insert(1, Tree(6))
>>> print(tree_copy.pretty())
      1
  ┌───┴┬──┐
  2    6  5
 ┌┴─┐
 3  4

# delete node 4
>>> del tree_copy[0][1]
>>> print(tree_copy.pretty())
    1
 ┌──┼──┐
 2  6  5
 │
 3
```

**Note**: Python lists are not optimized for insertion/deletion performance, so you should use this approach sparingly. It is better to build your tree structure up-front and change it as little as possible.

### Functional tree operations

`Tree` exposes a variety of operations from functional programming, which allow you to manipulate trees in a consistent way regardless of the kind of data they contain. We'll go through a few examples.

#### **Map**: apply a function element-wise

The `map` method takes a function and applies it to each element of the tree, preserving the tree structure.

```python
def square(x):
    """Take the square of a number."""
    return x ** 2

# square the value of each node
>>> tree_squared = tree.map(square)
>>> print(tree_squared.pretty())
     1
  ┌──┴──┐
  4     25
 ┌┴─┐
 9  16

# equivalently, as shorthand we can use a lambda expression
>>> tree.map(lambda x: x ** 2) == tree_squared
True
```

For convenience there is also a `leaf_map` method, which applies the function only to leaf nodes, and an `internal_map` method, which applies the function only to internal (non-leaf) nodes.

```python
>>> print(tree.leaf_map(square).pretty())
     1
  ┌──┴──┐
  2     25
 ┌┴─┐
 9  16

>>> print(tree.internal_map(square).pretty())
    1
  ┌─┴──┐
  4    5
 ┌┴─┐
 3  4
```

#### **Reduce**: combine all nodes together

The `reduce` method takes a function with two arguments used to combine two values into a single value. It recursively applies the operation to combine all nodes into one value.

The input function should take two values of the same type and return a value of that type. The function does not have to be associative or commutative, but `reduce` will be more "well-behaved" if it is (i.e. the result will not depend on the arrangement of nodes in the tree).

```python
from operator import add, mul

# add all the nodes together
>>> tree.reduce(add)
15

# multiply all the nodes together
>>> tree.reduce(mul)
120

# concatenate all the nodes together (as strings)
# NOTE: this operation is associative but *not* commutative, so order matters
>>> tree.map(str).reduce(add)
'12345'
```

#### **Scan**: partial reduce over each subtree

The `scan` method performs an operation which replaces each node in the tree with the result of running `reduce` with some function over that node's subtree. In effect this creates a tree of "partial" reductions.

For instance, if the provided function is `add`, this will produce the tree of "partial sums."

```python
from operator import add

# original tree
>>> print(tree.pretty())
    1
  ┌─┴──┐
  2    5
 ┌┴─┐
 3  4

# tree of partial sums
>>> print(tree.scan(add).pretty())
    15
  ┌─┴──┐
  9    5
 ┌┴─┐
 3  4
```

#### **Fold**: general purpose bottom-up recursion

The `fold` method is a general purpose construct for performing bottom-up recursion on a tree. The input is a function `f` taking two arguments, a parent node and a list of already-processed children. Starting at the root node, `tree.fold(f)` does the following:

1. Recursively call `subtree.fold(f)` on each of the child subtrees.
2. Return the result of `f(parent_node, processed_subtrees)`.

In functional programming this is also known as a *tree catamorphism*. It captures the pattern of building some value from the bottom of the tree upward. This means that nodes can pass information up to their ancestors, but not vice versa.

Fold is in fact a generalization of all the previous patterns discussed in this section; you can actually express `map`, `reduce`, and `scan` all in terms of `fold`!

Here is an example of using `fold` to modify a tree so that each node is converted to a pair `(node, num_descendants)`, where `node` is the original node's value, and `num_descendants` is the total number of descendants of that node.

```python
def f(node, children):
    # add the number of children to the total number of childrens' descendants
    num_descendants = len(children) + sum(child.node[1] for child in children)
    # return a new tree whose root node includes the number of descendants
    return Tree((node, num_descendants), children)

>>> tree_with_descendants = tree.fold(f)

>>> print(tree_with_descendants.pretty())
           (1, 4)
       ┌─────┴─────┐
     (2, 2)      (5, 0)
   ┌───┴───┐
 (3, 0)  (4, 0)
```

## License

This library is open-source and licensed under the [MIT License](LICENSE).

Contributions are welcome!
