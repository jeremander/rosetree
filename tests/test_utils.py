import pytest

from rosetree.utils import merge_dicts


@pytest.mark.parametrize(['d1', 'd2', 'output'], [
    ({}, {}, {}),
    ({'a': 1}, {}, {'a': 1}),
    ({'a': 1}, {'b': 2}, {'a': 1, 'b': 2}),
    ({'a': 1}, {'a': 2}, None),
    ({'a': 1}, {'a': 2, 'b': 3}, None),
])
def test_merge_dicts(d1, d2, output):
    if output is None:
        with pytest.raises(KeyError, match='Duplicate keys found'):
            _ = merge_dicts(d1, d2)
    else:
        assert merge_dicts(d1, d2) == output
