from collections.abc import Iterable
from itertools import accumulate, chain
import math
from typing import Hashable, Mapping, TypeVar


K = TypeVar('K', bound=Hashable)
V = TypeVar('V')


def merge_dicts(d1: Mapping[K, V], d2: Mapping[K, V]) -> dict[K, V]:
    """Merges two dictionaries together, raising a KeyError if any key appears in both dictionaries."""
    common_keys = set(d1) & set(d2)
    if common_keys:
        raise KeyError(f'Duplicate keys found: {common_keys}')
    # Merge the dictionaries
    return {**d1, **d2}

def cumsums(xs: Iterable[float]) -> list[float]:
    """Computes cumulative sums of a sequence of numbers, starting with 0."""
    return list(accumulate(chain([0.0], xs)))

def round_significant_figures(x: float, sigfigs: int) -> float:
    """Rounds a number to some number of significant figures."""
    if x == 0:
        return 0
    abs_x = abs(x)
    order = math.floor(math.log10(abs_x))
    factor = 10 ** (sigfigs - 1 - order)
    return math.copysign(round(abs_x * factor) / factor, x)

def make_percent(x: float) -> str:
    """Converts a float to a readable percentage."""
    pct = 100.0 * x
    pct = round_significant_figures(pct, 2)
    if pct == int(pct):
        pct = int(pct)
    pct_str = f'{pct:f}'.rstrip('0').rstrip('.')
    return f'{pct_str}%'
