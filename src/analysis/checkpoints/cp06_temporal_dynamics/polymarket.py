"""CP06 — Temporal Dynamics: Polymarket cross-platform check.

Tracks monthly trade volume on Polymarket via block timestamps as a proxy
for market maturation and growing participation over time.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.common.analysis import Analysis, AnalysisOutput


class PolymarketTemporalDynamics(Analysis):
    """Polymarket monthly trade volume trend as a market maturation proxy."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        legacy_trades_dir: Path | str | None = None,
        blocks_dir: Path | str | None = None,
    ):
        super().__init__(
            name="polymarket_cp06_temporal_dynamics",
            description="Polymarket monthly volume trend: market maturation proxy",
        )
        base = Path(__file__).parent.parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base / "data" / "polymarket" / "trades")
        self.legacy_dir = Path(legacy_trades_dir or base / "data" / "polymarket" / "legacy_trades")
        self.blocks_dir = Path(blocks_dir or base / "data" / "polymarket" / "blocks")

    def run(self) -> AnalysisOutput:
        con = duckdb.connect()

        with self.progress("Joining CTF trades with block timestamps"):
            ctf = con.execute(f"""
                SELECT
                    DATE_TRUNC('month', b.timestamp::TIMESTAMP) AS month,
                    COUNT(*)                                     AS n_trades
                FROM '{self.trades_dir}/*.parquet' t
                JOIN '{self.blocks_dir}/*.parquet' b ON t.block_number = b.block_number
                GROUP BY month
            """).df()

        with self.progress("Joining legacy trades with block timestamps"):
            leg = con.execute(f"""
                SELECT
                    DATE_TRUNC('month', b.timestamp::TIMESTAMP) AS month,
                    COUNT(*)                                     AS n_trades
                FROM '{self.legacy_dir}/*.parquet' t
                JOIN '{self.blocks_dir}/*.parquet' b ON t.block_number = b.block_number
                GROUP BY month
            """).df()

        monthly = (
            pd.concat([ctf, leg])
            .groupby("month")["n_trades"]
            .sum()
            .reset_index()
            .sort_values("month")
        )
        monthly["month"] = pd.to_datetime(monthly["month"])
        monthly["tau"] = np.arange(len(monthly))
        monthly = monthly.dropna()

        result = stats.linregress(monthly["tau"], monthly["n_trades"])
        gamma1 = float(result.slope)
        gamma1_p = float(result.pvalue)
        monthly["fitted"] = result.intercept + gamma1 * monthly["tau"]

        summary = pd.DataFrame([
            {"metric": "gamma1_slope",  "value": round(gamma1, 2)},
            {"metric": "gamma1_p",      "value": round(gamma1_p, 6)},
            {"metric": "n_months",      "value": len(monthly)},
        ])

        from src.common.plot_style import new_fig, clean_ax, stat_box, sig_stars, GREEN, RED

        fig, ax = new_fig(1, 1, figsize=(13, 5.5),
                          suptitle="Polymarket — Monthly Trade Volume (Market Maturation)")
        stars = sig_stars(gamma1_p)
        ax.bar(monthly["month"], monthly["n_trades"] / 1_000,
               width=25, color=GREEN, alpha=0.70, linewidth=0, label="Monthly Trades (CTF + Legacy)")
        ax.plot(monthly["month"], monthly["fitted"] / 1_000,
                color=RED, lw=2.2, label=f"OLS trend  γ₁ = {gamma1:,.0f} trades/mo")
        clean_ax(ax,
                 xlabel="Month",
                 ylabel="Trades (thousands)",
                 title="",
                 zero_h=False)
        ax.legend(loc="upper left")
        stat_box(ax, f"γ₁ = {gamma1:,.0f} trades/mo\np = {gamma1_p:.4f}  {stars}",
                 loc="lower right")
        fig.tight_layout()

        return AnalysisOutput(figure=fig, data=summary)
