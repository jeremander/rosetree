import pytest

from rosetree.utils import make_percent, merge_dicts, round_significant_figures


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

@pytest.mark.parametrize(['x', 'sigfigs', 'output'], [
    (0, 0, 0),
    (0, 3, 0),
    (2.7, 1, 3),
    (2.7, 2, 2.7),
    (2.7, 5, 2.7),
])
def test_round_significant_figures(x, sigfigs, output):
    assert round_significant_figures(x, sigfigs) == output

@pytest.mark.parametrize(['x', 'percent'], [
    (0, '0%'),
    (0.000001, '0.0001%'),
    (0.0000012, '0.00012%'),
    (0.00000125, '0.00012%'),  # NOTE: this should be 0.00013%
    (0.00000126, '0.00013%'),
    (0.0001, '0.01%'),
    (0.001, '0.1%'),
    (0.01, '1%'),
    (0.1, '10%'),
    (0.5, '50%'),
    (0.99, '99%'),
    (0.994, '99%'),
    (0.995, '100%'),
    (0.999, '100%'),
    (1, '100%'),

])
def test_make_percent(x, percent):
    assert make_percent(x) == percent
