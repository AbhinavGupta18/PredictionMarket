"""Shared metrics helpers for prediction market microstructure analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def sharpe_ratio(returns: pd.Series | np.ndarray, annualize: bool = False) -> float:
    """Compute the Sharpe ratio (mean / std) of a return series."""
    r = np.asarray(returns, dtype=float)
    std = r.std(ddof=1)
    if std == 0 or np.isnan(std):
        return float("nan")
    sr = r.mean() / std
    if annualize:
        sr *= np.sqrt(252)
    return float(sr)


def volume_weighted_return(returns: np.ndarray, volumes: np.ndarray) -> float:
    """Compute the volume-weighted average return."""
    volumes = np.asarray(volumes, dtype=float)
    returns = np.asarray(returns, dtype=float)
    total = volumes.sum()
    if total == 0:
        return float("nan")
    return float((returns * volumes).sum() / total)


def one_sample_ttest(values: np.ndarray, mu0: float = 0.0) -> tuple[float, float]:
    """Two-sided one-sample t-test. Returns (t_stat, p_value)."""
    t, p = stats.ttest_1samp(values, mu0, nan_policy="omit")
    return float(t), float(p)


def taker_pnl_per_contract(taker_side: str, yes_price_cents: float, result: str) -> float:
    """Compute per-contract PnL for the taker (fractional dollars).

    Returns value in [-1, 1] range.
    - result: 'yes' or 'no'
    - yes_price_cents: 0-100
    """
    result_num = 1.0 if result == "yes" else 0.0
    if taker_side == "yes":
        return result_num - yes_price_cents / 100.0
    else:
        return (1.0 - result_num) - (100.0 - yes_price_cents) / 100.0


def maker_pnl_per_contract(taker_side: str, yes_price_cents: float, result: str) -> float:
    """Compute per-contract PnL for the maker (exact negative of taker PnL)."""
    return -taker_pnl_per_contract(taker_side, yes_price_cents, result)


# SQL snippets reused across checkpoint analyses
TAKER_PNL_SQL = """
    CASE
        WHEN t.taker_side = 'yes' THEN
            (CASE WHEN m.result = 'yes' THEN 1.0 ELSE 0.0 END) - t.yes_price / 100.0
        ELSE
            (CASE WHEN m.result = 'no'  THEN 1.0 ELSE 0.0 END) - t.no_price  / 100.0
    END
"""

MAKER_PNL_SQL = """
    CASE
        WHEN t.taker_side = 'yes' THEN
            t.yes_price / 100.0 - (CASE WHEN m.result = 'yes' THEN 1.0 ELSE 0.0 END)
        ELSE
            t.no_price  / 100.0 - (CASE WHEN m.result = 'no'  THEN 1.0 ELSE 0.0 END)
    END
"""

TAKER_PRICE_SQL = """
    CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END
"""

MAKER_PRICE_SQL = """
    CASE WHEN t.taker_side = 'yes' THEN t.no_price ELSE t.yes_price END
"""
