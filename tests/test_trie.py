import pytest

from rosetree import Trie


def test_contains():
    trie = Trie.from_sequence('123')
    assert '123' in trie
    assert ('1', '2', '3') in trie
    assert 123 not in trie  # type: ignore[comparison-overlap]
    assert (1, 2, 3) not in trie  # type: ignore[comparison-overlap]

@pytest.mark.parametrize('strings', [
    [],
    [''],
    ['a'],
    ['abc'],
    ['a', 'b', 'c'],
    ['', 'a', 'aa', 'aaa'],
    ['a', 'ab', 'abc'],
    ['', 'a', 'aa', 'ab'],
    ['', 'a', 'bc'],
    ['', 'a', 'abc'],
])
def test_trie_properties(strings):
    trie = Trie.from_sequences(strings)
    assert len(trie) == len(strings)
    # check the set of iterated tuples matches the set of strings
    tups = list(trie)
    assert all(isinstance(tup, tuple) for tup in tups)
    assert sorted(''.join(tup) for tup in tups) == sorted(strings)
    # check membership (can be either strings or tuples)
    assert all(tup in trie for tup in tups)
    assert all(s in trie for s in strings)

def test_subtrie():
    trie = Trie.from_sequences(['', 'a', 'bc', '1'])
    # root is a member
    assert trie.member
    # subtrie with empty sequence is the original trie
    assert trie.subtrie('') is trie
    # proper subtries
    assert trie.subtrie('a') == Trie(member=True)
    assert trie.subtrie('b') == Trie(member=False, children={'c': Trie(member=True)})
    assert trie.subtrie('bc') == Trie(member=True)
    for sym in ['c', 'ab', 'abc']:
        assert trie.subtrie(sym) is None
    assert trie.subtrie('1') == Trie(member=True)
    with pytest.raises(TypeError, match='not subscriptable'):
        _ = trie.subtrie(1)
    assert trie.subtrie((1,)) is None
