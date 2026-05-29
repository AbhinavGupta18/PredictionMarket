"""CP03 — Maker-Taker Wealth Transfer: Kalshi.

Computes trade-level PnL for makers and takers, aggregates volume-weighted
average returns, and tests H0: Maker return = 0.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.common.analysis import Analysis, AnalysisOutput
from src.common.metrics import MAKER_PNL_SQL, TAKER_PNL_SQL, TAKER_PRICE_SQL


class KalshiMakerTakerPnl(Analysis):
    """Maker-Taker PnL: volume-weighted returns and t-test."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="kalshi_cp03_maker_taker_pnl",
            description="Maker vs taker wealth transfer: PnL, volume-weighted returns, t-test",
        )
        base = Path(__file__).parent.parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base / "data" / "kalshi" / "markets")

    def run(self) -> AnalysisOutput:
        con = duckdb.connect()

        with self.progress("Computing trade-level PnL"):
            df = con.execute(f"""
                WITH resolved AS (
                    SELECT ticker, result
                    FROM '{self.markets_dir}/*.parquet'
                    WHERE status = 'finalized' AND result IN ('yes','no')
                )
                SELECT
                    DATE_TRUNC('month', t.created_time) AS month,
                    {TAKER_PRICE_SQL}                   AS taker_price_cents,
                    {TAKER_PNL_SQL}                     AS taker_pnl,
                    {MAKER_PNL_SQL}                     AS maker_pnl,
                    t.count                             AS contracts,
                    t.count * ({TAKER_PRICE_SQL}) / 100.0 AS taker_notional
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN resolved m ON t.ticker = m.ticker
            """).df()

        # Volume-weighted returns
        taker_vwr = float(
            (df["taker_pnl"] * df["contracts"]).sum() / df["contracts"].sum()
        )
        maker_vwr = float(
            (df["maker_pnl"] * df["contracts"]).sum() / df["contracts"].sum()
        )

        # Aggregate PnL
        total_taker_pnl = float((df["taker_pnl"] * df["contracts"]).sum())
        total_maker_pnl = float((df["maker_pnl"] * df["contracts"]).sum())

        # T-test on monthly VWR series: avoids inflating n with correlated trade rows
        monthly_maker_vwr = (
            df.groupby("month")
            .apply(lambda g: (g["maker_pnl"] * g["contracts"]).sum() / g["contracts"].sum())
            .dropna()
            .values
        )
        maker_t, maker_p = stats.ttest_1samp(monthly_maker_vwr, 0.0)

        # By price bucket (10-cent bins)
        df["bucket"] = (df["taker_price_cents"] // 10) * 10
        bucket_stats = (
            df.groupby("bucket")
            .apply(lambda g: pd.Series({
                "n_trades":    len(g),
                "n_contracts": g["contracts"].sum(),
                "taker_vwr":   (g["taker_pnl"] * g["contracts"]).sum() / g["contracts"].sum(),
                "maker_vwr":   (g["maker_pnl"] * g["contracts"]).sum() / g["contracts"].sum(),
            }))
            .reset_index()
        )

        # Summary table
        summary = pd.DataFrame([
            {"metric": "taker_vw_return",    "value": round(taker_vwr, 6),       "unit": "fraction"},
            {"metric": "maker_vw_return",    "value": round(maker_vwr, 6),       "unit": "fraction"},
            {"metric": "total_taker_pnl",    "value": round(total_taker_pnl, 2), "unit": "contracts"},
            {"metric": "total_maker_pnl",    "value": round(total_maker_pnl, 2), "unit": "contracts"},
            {"metric": "maker_t_stat",       "value": round(float(maker_t), 4),  "unit": ""},
            {"metric": "maker_p_value",      "value": round(float(maker_p), 6),  "unit": ""},
            {"metric": "n_trades",           "value": len(df),                   "unit": ""},
            {"metric": "n_contracts",        "value": int(df["contracts"].sum()), "unit": ""},
        ])

        fig = self._make_figure(bucket_stats, taker_vwr, maker_vwr, maker_t, maker_p)
        return AnalysisOutput(figure=fig, data=summary)

    def _make_figure(
        self,
        bucket: pd.DataFrame,
        taker_vwr: float,
        maker_vwr: float,
        maker_t: float,
        maker_p: float,
    ) -> plt.Figure:
        from src.common.plot_style import (
            new_fig, clean_ax, stat_box, sig_stars, BLUE, GREEN, RED, GRAY,
        )

        fig, axes = new_fig(1, 2, suptitle="Kalshi — Maker-Taker Wealth Transfer")
        stars = sig_stars(maker_p)

        # ── Left: VWR by price bucket ─────────────────────────────────────────
        x = np.arange(len(bucket))
        w = 0.37
        axes[0].bar(x - w / 2, bucket["taker_vwr"] * 100, w,
                    label="Taker", color=RED, alpha=0.85, linewidth=0)
        axes[0].bar(x + w / 2, bucket["maker_vwr"] * 100, w,
                    label="Maker", color=GREEN, alpha=0.85, linewidth=0)
        clean_ax(axes[0],
                 xlabel="Taker Price Bucket",
                 ylabel="Volume-Weighted Return (pp)",
                 title="Maker vs Taker Returns by Price Bucket")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([f"{int(b)}¢" for b in bucket["bucket"]], fontsize=9)
        axes[0].legend()

        # ── Right: aggregate comparison ───────────────────────────────────────
        ax2 = axes[1]
        vals = [taker_vwr * 100, maker_vwr * 100]
        clrs = [RED, GREEN]
        bars = ax2.bar(["Taker", "Maker"], vals, color=clrs, width=0.42, alpha=0.9,
                       linewidth=0)
        ax2.bar_label(bars, fmt="%.3f pp", padding=4, fontsize=9.5)
        # Arrow showing wealth direction
        ax2.annotate(
            "", xy=(1, maker_vwr * 100), xytext=(0, taker_vwr * 100),
            xycoords=("data", "data"),
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.5),
        )
        clean_ax(ax2,
                 ylabel="Volume-Weighted Return (pp)",
                 title="Aggregate VW Returns")
        stat_box(ax2,
                 f"Maker monthly t-test\nt = {maker_t:.2f}, p = {maker_p:.4f} {stars}",
                 loc="lower right")

        fig.tight_layout()
        return fig
