"""Small statistics helpers for comparisons (plan Q.5): t-based CIs, paired differences.

ponytail: a Student-t table (df 1-30, 40, 60, 120, inf; interpolated in 1/df) instead of
scipy; exact enough for 95 % intervals.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final

import numpy as np

__all__ = ["mean_ci", "paired_diff", "t_quantile"]

_T975: Final = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
    9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120,
    17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
    25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042, 40: 2.021, 60: 2.000,
    120: 1.980,
}  # fmt: skip
_T_INF: Final = 1.960


def t_quantile(df: int) -> float:
    """Two-sided 95 % Student-t quantile ``t_{0.975, df}`` (``df >= 1``)."""
    if df < 1:
        raise ValueError("df must be >= 1")
    if df in _T975:
        return _T975[df]
    keys = sorted(_T975)
    if df > keys[-1]:
        lo_df, lo_t, hi_inv, hi_t = keys[-1], _T975[keys[-1]], 0.0, _T_INF
    else:
        lo_df = max(k for k in keys if k < df)
        hi_df = min(k for k in keys if k > df)
        lo_t, hi_t, hi_inv = _T975[lo_df], _T975[hi_df], 1.0 / hi_df
    w = (1.0 / lo_df - 1.0 / df) / (1.0 / lo_df - hi_inv)
    return lo_t + w * (hi_t - lo_t)


def mean_ci(values: Sequence[float]) -> tuple[float, float, float, float]:
    """``(mean, std, ci_lo, ci_hi)``; NaN values are ignored; the CI is NaN for n < 2."""
    x = np.asarray([v for v in values if not math.isnan(v)], dtype=np.float64)
    if x.size == 0:
        return math.nan, math.nan, math.nan, math.nan
    mean = float(x.mean())
    if x.size < 2:
        return mean, math.nan, math.nan, math.nan
    std = float(x.std(ddof=1))
    half = t_quantile(x.size - 1) * std / math.sqrt(x.size)
    return mean, std, mean - half, mean + half


def paired_diff(
    variant: Sequence[float], baseline: Sequence[float]
) -> tuple[float, float, float, int]:
    """Paired ``variant - baseline`` over matching positions (seeds): ``(mean, ci_lo,
    ci_hi, n)``; pairs with a NaN are dropped."""
    pairs = [
        (v, b)
        for v, b in zip(variant, baseline, strict=True)
        if not (math.isnan(v) or math.isnan(b))
    ]
    mean, _std, lo, hi = mean_ci([v - b for v, b in pairs])
    return mean, lo, hi, len(pairs)
