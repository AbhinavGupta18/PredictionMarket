"""CP02 — Favorite-Longshot Bias: Polymarket cross-platform check.

Replicates the calibration deviation analysis on Polymarket CTF trades
to test whether FLSB is platform-specific or a universal prediction-market phenomenon.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.common.analysis import Analysis, AnalysisOutput

BIN_WIDTH = 5


class PolymarketLongshotBias(Analysis):
    """Favorite-Longshot Bias on Polymarket: calibration deviation by price bucket."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="polymarket_cp02_longshot_bias",
            description="Polymarket FLSB: empirical win rate vs implied probability by bucket",
        )
        base = Path(__file__).parent.parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base / "data" / "polymarket" / "trades")
        self.markets_dir = Path(markets_dir or base / "data" / "polymarket" / "markets")

    def run(self) -> AnalysisOutput:
        con = duckdb.connect()

        with self.progress("Resolving market outcomes"):
            markets_df = con.execute(f"""
                SELECT id, clob_token_ids, outcome_prices
                FROM '{self.markets_dir}/*.parquet'
                WHERE closed = true
                  AND clob_token_ids IS NOT NULL
                  AND outcome_prices IS NOT NULL
            """).df()

        token_won: dict[str, bool] = {}
        for _, row in markets_df.iterrows():
            try:
                prices = json.loads(row["outcome_prices"])
                tokens = json.loads(row["clob_token_ids"])
                if len(prices) != 2 or len(tokens) != 2:
                    continue
                p0, p1 = float(prices[0]), float(prices[1])
                if p0 > 0.99 and p1 < 0.01:
                    token_won[tokens[0]] = True
                    token_won[tokens[1]] = False
                elif p0 < 0.01 and p1 > 0.99:
                    token_won[tokens[0]] = False
                    token_won[tokens[1]] = True
            except (json.JSONDecodeError, ValueError, TypeError, IndexError):
                continue

        con.execute("CREATE TABLE token_res (token_id VARCHAR, won BOOLEAN)")
        con.executemany("INSERT INTO token_res VALUES (?,?)", list(token_won.items()))

        with self.progress("Computing price bucket calibration"):
            df = con.execute(f"""
                WITH raw AS (
                    SELECT
                        CASE
                            WHEN t.maker_asset_id = '0'
                            THEN ROUND(100.0 * t.maker_amount / t.taker_amount)
                            ELSE ROUND(100.0 * t.taker_amount / t.maker_amount)
                        END AS price_cents,
                        tr.won
                    FROM '{self.trades_dir}/*.parquet' t
                    INNER JOIN token_res tr ON (
                        CASE WHEN t.maker_asset_id='0' THEN t.taker_asset_id
                             ELSE t.maker_asset_id END = tr.token_id
                    )
                    WHERE t.taker_amount > 0 AND t.maker_amount > 0
                )
                SELECT
                    FLOOR(price_cents / {BIN_WIDTH}) * {BIN_WIDTH}          AS bin_low,
                    FLOOR(price_cents / {BIN_WIDTH}) * {BIN_WIDTH} + {BIN_WIDTH}/2.0 AS bin_mid,
                    COUNT(*)                                                  AS n_trades,
                    AVG(price_cents) / 100.0                                  AS avg_implied_prob,
                    AVG(won::INT)                                             AS empirical_win_rate,
                    AVG(won::INT) - AVG(price_cents) / 100.0                  AS delta_b,
                    STDDEV_POP(won::INT)                                      AS std_won,
                    COUNT(*)                                                  AS n
                FROM raw
                WHERE price_cents BETWEEN 1 AND 99
                GROUP BY bin_low, bin_mid
                HAVING COUNT(*) >= 30
                ORDER BY bin_low
            """).df()

        df["se"] = df["std_won"] / np.sqrt(df["n"])
        df["t_stat"] = df["delta_b"] / df["se"].replace(0, np.nan)
        df["p_value"] = df["t_stat"].apply(
            lambda t: float(2 * stats.norm.sf(abs(t))) if not np.isnan(t) else np.nan
        )

        low_prob = df[df["avg_implied_prob"] < 0.25]
        if len(low_prob) > 0:
            flsb_t, flsb_p = stats.ttest_1samp(low_prob["delta_b"].dropna(), 0.0)
        else:
            flsb_t, flsb_p = float("nan"), float("nan")

        fig = self._make_figure(df, float(flsb_t), float(flsb_p))
        return AnalysisOutput(figure=fig, data=df)

    def _make_figure(self, df: pd.DataFrame, flsb_t: float, flsb_p: float) -> plt.Figure:
        from src.common.plot_style import (
            new_fig, clean_ax, stat_box, sig_stars, shade_h, bar_colors,
            BLUE, GREEN, RED, GRAY, CMAP_DIV,
        )

        fig, axes = new_fig(1, 2, suptitle="Polymarket — Favorite-Longshot Bias")
        stars = sig_stars(flsb_p)

        ax = axes[0]
        shade_h(ax, 0, 25,  color=RED,   alpha=0.07)
        shade_h(ax, 75, 100, color=GREEN, alpha=0.07)
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
                 xlabel="Implied Probability (%)",
                 ylabel="Empirical Win Rate (%)",
                 title="Calibration Curve — CTF Token Buyers",
                 zero_h=False)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8.5)

        ax2 = axes[1]
        colors = bar_colors(df["delta_b"], pos_color=BLUE, neg_color=RED)
        ax2.bar(df["bin_mid"], df["delta_b"] * 100,
                width=BIN_WIDTH * 0.82,
                color=colors, edgecolor="white", linewidth=0.3, alpha=0.9)
        shade_h(ax2, 0, 25,  color=RED,   alpha=0.06)
        shade_h(ax2, 75, 100, color=GREEN, alpha=0.06)
        clean_ax(ax2,
                 xlabel="Price Bucket Midpoint (%)",
                 ylabel="δ_b  =  f(b) − P̄_b  (pp)",
                 title="Calibration Deviation δ_b by Bucket",
                 zero_h=True)
        stat_box(ax2,
                 f"Low-price buckets (<25%)\nt = {flsb_t:.2f}, p = {flsb_p:.4f} {stars}",
                 loc="upper right")

        fig.tight_layout()
        return fig
