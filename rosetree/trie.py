from __future__ import annotations

from collections.abc import Collection, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Hashable, Optional, TypeVar

from typing_extensions import Self


T = TypeVar('T', bound=Hashable)


@dataclass
class Trie(Collection[Sequence[T]]):
    """A simple trie (prefix tree) data structure, representing a set of sequences.
    Sequence elements must be hashable."""

    # indicates whether this prefix is a member of the set, or just a proper prefix
    member: bool = False
    # dict mapping from symbols to subtries
    children: dict[T, Trie[T]] = field(default_factory=dict)

    def __len__(self) -> int:
        """Returns the number of elements in the Trie."""
        return int(self.member) + sum(len(child) for child in self.children.values())

    def __iter__(self) -> Iterator[tuple[T, ...]]:
        """Iterates through elements of the Trie with pre-order traversal."""
        def _iter_with_prefix(prefix: tuple[T, ...], trie: Trie[T]) -> Iterator[tuple[T, ...]]:
            if trie.member:
                yield prefix
            if len(trie) > 0:
                for (parent, subtrie) in trie.children.items():
                    # assume the sequence can be concatenated with +
                    yield from _iter_with_prefix(prefix + (parent,), subtrie)
        return _iter_with_prefix((), self)

    def __contains__(self, seq: object) -> bool:
        """Returns True if the given sequence is an element of the Trie."""
        if not isinstance(seq, Sequence):
            return False
        if not seq:
            return self.member
        head, tail = seq[0], seq[1:]
        return (head in self.children) and (tail in self.children[head])

    def add(self, seq: Sequence[T]) -> None:
        """Adds a sequence to the Trie."""
        if seq:
            head, tail = seq[0], seq[1:]
            if head not in self.children:
                self.children[head] = type(self)()
            self.children[head].add(tail)
        else:
            self.member = True

    @classmethod
    def from_sequences(cls, sequences: Iterable[Sequence[T]]) -> Self:
        """Constructs a Trie from sequences of symbols."""
        trie = cls()
        for seq in sequences:
            trie.add(seq)
        return trie

    @classmethod
    def from_sequence(cls, sequence: Sequence[T]) -> Self:
        """Constructs a Trie from a single sequence of symbols."""
        return cls.from_sequences([sequence])

    def subtrie(self, prefix: Sequence[T]) -> Optional[Trie[T]]:
        """If the given prefix is in the Trie, returns the subtrie rooted by that prefix.
        Otherwise, returns None."""
        if not prefix:
            return self
        head, tail = prefix[0], prefix[1:]
        if head in self.children:
            return self.children[head].subtrie(tail)
        return None
