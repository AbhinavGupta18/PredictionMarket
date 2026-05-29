"""CP09 — Volume Demand Elasticity & Affirmative Preference: Kalshi.

Computes the YES/NO volume ratio across taker trades and tests whether it
deviates significantly from 0.50 (the rational neutral baseline).
Stratifies by category to detect domain-specific affirmative bias.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.kalshi.util.categories import category_sql, get_group
from src.common.analysis import Analysis, AnalysisOutput


class KalshiVolumeDemandElasticity(Analysis):
    """Affirmative preference: YES/NO volume ratio and binomial test."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="kalshi_cp09_volume_demand_elasticity",
            description="YES/NO taker volume ratio: test for affirmative bias away from 0.50",
        )
        base = Path(__file__).parent.parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base / "data" / "kalshi" / "markets")

    def run(self) -> AnalysisOutput:
        con = duckdb.connect()

        with self.progress("Aggregating YES/NO capital by category"):
            df = con.execute(f"""
                WITH mkts AS (
                    SELECT ticker, event_ticker
                    FROM '{self.markets_dir}/*.parquet'
                ),
                trades AS (
                    SELECT
                        {category_sql("m")} AS category,
                        taker_side,
                        -- Capital committed = price × count
                        CASE WHEN taker_side = 'yes'
                             THEN yes_price * t.count / 100.0
                             ELSE no_price  * t.count / 100.0
                        END AS capital
                    FROM '{self.trades_dir}/*.parquet' t
                    LEFT JOIN mkts m ON t.ticker = m.ticker
                )
                SELECT
                    category,
                    SUM(CASE WHEN taker_side = 'yes' THEN capital ELSE 0 END) AS yes_capital,
                    SUM(CASE WHEN taker_side = 'no'  THEN capital ELSE 0 END) AS no_capital,
                    SUM(capital)                                               AS total_capital
                FROM trades
                GROUP BY category
            """).df()

        df["group"] = df["category"].apply(get_group)

        group_agg = (
            df.groupby("group")[["yes_capital", "no_capital", "total_capital"]]
            .sum()
            .reset_index()
        )
        group_agg["yes_share"] = group_agg["yes_capital"] / group_agg["total_capital"]

        # Global ratio
        total_yes = float(df["yes_capital"].sum())
        total_no  = float(df["no_capital"].sum())
        total_all = total_yes + total_no
        global_ratio = total_yes / total_all

        # Binomial test: H0 ratio = 0.50; contracts = total taker trades
        total_trades = con.execute(f"""
            SELECT
                SUM(CASE WHEN taker_side = 'yes' THEN 1 ELSE 0 END) AS yes_n,
                COUNT(*) AS n
            FROM '{self.trades_dir}/*.parquet'
        """).fetchone()
        yes_n, n = int(total_trades[0]), int(total_trades[1])
        binom_result = stats.binomtest(yes_n, n, p=0.5, alternative="two-sided")
        binom_p = float(binom_result.pvalue)

        summary = pd.DataFrame([
            {"metric": "global_yes_capital_share", "value": round(global_ratio, 4)},
            {"metric": "total_yes_capital",        "value": round(total_yes, 2)},
            {"metric": "total_no_capital",         "value": round(total_no, 2)},
            {"metric": "yes_n_trades",             "value": yes_n},
            {"metric": "total_n_trades",           "value": n},
            {"metric": "binomial_p_value",         "value": round(binom_p, 8)},
        ])

        fig = self._make_figure(group_agg, global_ratio, binom_p)
        return AnalysisOutput(figure=fig, data=summary)

    def _make_figure(
        self,
        group_agg: pd.DataFrame,
        global_ratio: float,
        binom_p: float,
    ) -> plt.Figure:
        from src.common.plot_style import (
            new_fig, clean_ax, stat_box, sig_stars, group_color,
            GREEN, RED, GRAY,
        )

        fig, axes = new_fig(1, 2, suptitle="Kalshi — Affirmative Demand Bias")
        stars = sig_stars(binom_p)
        top = group_agg.sort_values("total_capital", ascending=False).head(10)

        # ── Left: YES share by category (diverging from 50%) ─────────────────
        ax = axes[0]
        grp_list   = list(reversed(top["group"].tolist()))
        share_list = list(reversed((top["yes_share"] * 100).tolist()))
        clrs = [GREEN if s > 50 else RED for s in share_list]
        ax.barh(grp_list, share_list, color=clrs, alpha=0.88, linewidth=0)
        ax.axvline(50, color=GRAY, lw=1.2, linestyle="--", label="50% neutral")
        ax.set_xlim(30, 70)
        clean_ax(ax,
                 xlabel="YES Capital Share (%)",
                 title="YES Taker Capital Share by Category",
                 zero_h=False)
        ax.legend(fontsize=8.5)

        # ── Right: global split with annotation ──────────────────────────────
        ax2 = axes[1]
        vals = [global_ratio * 100, (1 - global_ratio) * 100]
        bars = ax2.bar(["YES", "NO"], vals,
                       color=[GREEN, RED], width=0.44, alpha=0.9, linewidth=0)
        ax2.axhline(50, color=GRAY, lw=1.2, linestyle="--")
        ax2.bar_label(bars, fmt="%.2f%%", padding=4, fontsize=9.5)
        clean_ax(ax2,
                 ylabel="Capital Share (%)",
                 title="Global Taker Capital: YES vs NO",
                 zero_h=False)
        stat_box(ax2,
                 f"Binomial test (H₀: 50/50)\np = {binom_p:.2e}  {stars}",
                 loc="upper right")

        fig.tight_layout()
        return fig
