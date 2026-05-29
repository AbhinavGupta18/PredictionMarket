"""CP02 — Favorite-Longshot Bias (FLSB): Kalshi.

Groups trades into 5-cent price bins and tests whether low-probability
contracts exhibit systematically negative calibration deviation δ_b = f(b) - P̄_b.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.common.analysis import Analysis, AnalysisOutput


class KalshiLongshotBias(Analysis):
    """Favorite-Longshot Bias: empirical win rate vs implied probability."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
        bin_width: int = 5,
    ):
        super().__init__(
            name="kalshi_cp02_longshot_bias",
            description="FLSB calibration curve: empirical frequency vs implied probability",
        )
        base = Path(__file__).parent.parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base / "data" / "kalshi" / "markets")
        self.bin_width = bin_width

    def run(self) -> AnalysisOutput:
        con = duckdb.connect()

        with self.progress("Computing bucket-level win rates"):
            df = con.execute(f"""
                WITH resolved AS (
                    SELECT ticker, result
                    FROM '{self.markets_dir}/*.parquet'
                    WHERE status = 'finalized' AND result IN ('yes','no')
                ),
                trades AS (
                    SELECT
                        t.yes_price                                             AS price_cents,
                        FLOOR(t.yes_price / {self.bin_width}) * {self.bin_width} AS bucket,
                        CASE WHEN t.taker_side = 'yes' AND m.result = 'yes' THEN 1
                             WHEN t.taker_side = 'yes' AND m.result = 'no'  THEN 0
                             ELSE NULL END                                       AS taker_yes_won,
                        t.count                                                 AS contracts
                    FROM '{self.trades_dir}/*.parquet' t
                    INNER JOIN resolved m ON t.ticker = m.ticker
                    WHERE t.taker_side = 'yes'
                )
                SELECT
                    bucket                                      AS bin_low,
                    bucket + {self.bin_width} / 2.0             AS bin_mid,
                    COUNT(*)                                    AS n_trades,
                    SUM(contracts)                              AS n_contracts,
                    AVG(price_cents) / 100.0                    AS avg_implied_prob,
                    AVG(taker_yes_won)                          AS empirical_win_rate,
                    AVG(taker_yes_won) - AVG(price_cents)/100.0 AS delta_b,
                    STDDEV_POP(taker_yes_won)                   AS std_won,
                    COUNT(*)                                    AS n
                FROM trades
                WHERE taker_yes_won IS NOT NULL
                GROUP BY bucket
                HAVING COUNT(*) >= 30
                ORDER BY bucket
            """).df()

        # One-sample t-test: delta_b = 0 vs delta_b != 0 per bucket
        # For low-probability buckets (< 25 cents) we expect delta_b < 0
        df["se"] = df["std_won"] / np.sqrt(df["n"])
        df["t_stat"] = df["delta_b"] / df["se"].replace(0, np.nan)
        df["p_value"] = df["t_stat"].apply(
            lambda t: float(2 * stats.norm.sf(abs(t))) if not np.isnan(t) else np.nan
        )

        # Low-probability buckets: price < 25 cents
        low_prob = df[df["avg_implied_prob"] < 0.25]
        if len(low_prob) > 0:
            flsb_t, flsb_p = stats.ttest_1samp(low_prob["delta_b"].dropna(), 0.0)
        else:
            flsb_t, flsb_p = float("nan"), float("nan")

        df.attrs["flsb_t_stat"] = float(flsb_t)
        df.attrs["flsb_p_value"] = float(flsb_p)

        fig = self._make_figure(df, flsb_t, flsb_p)
        return AnalysisOutput(figure=fig, data=df)

    def _make_figure(self, df: pd.DataFrame, flsb_t: float, flsb_p: float) -> plt.Figure:
        from src.common.plot_style import (
            new_fig, clean_ax, stat_box, sig_stars, shade_h, bar_colors,
            BLUE, GREEN, RED, GRAY, CMAP_DIV,
        )

        fig, axes = new_fig(1, 2, suptitle="Kalshi — Favorite-Longshot Bias")
        stars = sig_stars(flsb_p)

        # ── Left: calibration scatter ─────────────────────────────────────────
        ax = axes[0]
        shade_h(ax, 0, 25,  color=RED,   alpha=0.07, label="Longshot zone (<25¢)")
        shade_h(ax, 75, 100, color=GREEN, alpha=0.07, label="Favorite zone (>75¢)")
        sz = np.sqrt(df["n_trades"] / df["n_trades"].max()) * 220 + 18
        sc = ax.scatter(
            df["avg_implied_prob"] * 100,
            df["empirical_win_rate"] * 100,
            s=sz, c=df["delta_b"], cmap=CMAP_DIV, vmin=-0.06, vmax=0.06,
            edgecolors="#444", linewidths=0.3, alpha=0.88, zorder=3,
        )
        ax.plot([0, 100], [0, 100], "--", color=GRAY, lw=1.4, label="Perfect calibration")
        fig.colorbar(sc, ax=ax, label="δ_b (fraction)", shrink=0.85, pad=0.02)
        clean_ax(ax,
                 xlabel="Implied Probability (¢ = %)",
                 ylabel="Empirical Win Rate (%)",
                 title="Calibration Curve — Taker YES Bets",
                 zero_h=False)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8.5, loc="upper left")

        # ── Right: δ_b bar chart ──────────────────────────────────────────────
        ax2 = axes[1]
        colors = bar_colors(df["delta_b"], pos_color=BLUE, neg_color=RED)
        ax2.bar(df["bin_mid"], df["delta_b"] * 100,
                width=self.bin_width * 0.82,
                color=colors, edgecolor="white", linewidth=0.3, alpha=0.9)
        shade_h(ax2, 0, 25,  color=RED,   alpha=0.06)
        shade_h(ax2, 75, 100, color=GREEN, alpha=0.06)
        clean_ax(ax2,
                 xlabel="Price Bucket Midpoint (¢)",
                 ylabel="δ_b  =  f(b) − P̄_b  (pp)",
                 title="Calibration Deviation δ_b by Bucket",
                 zero_h=True)
        stat_box(ax2,
                 f"Low-price buckets (<25¢)\nt = {flsb_t:.2f}, p = {flsb_p:.4f} {stars}",
                 loc="upper right")

        fig.tight_layout()
        return fig
