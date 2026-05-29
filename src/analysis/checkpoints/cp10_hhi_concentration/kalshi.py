"""CP10 — Professionalization of Liquidity: Kalshi.

Computes Herfindahl-Hirschman Index (HHI) for contract-level volume concentration
as a proxy for liquidity consolidation over time.

Note: Individual maker identifiers are not available in the public trade data.
We use contract-ticker-level volume concentration as a proxy: high HHI across
individual contracts → volume is dominated by a few contracts.
We also track the top-1% contract volume share over quarterly cohorts.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.kalshi.util.categories import CATEGORY_SQL, get_group
from src.common.analysis import Analysis, AnalysisOutput


class KalshiHHIConcentration(Analysis):
    """HHI and top-1% concentration of trading volume across markets."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="kalshi_cp10_hhi_concentration",
            description="HHI market volume concentration + top-1% share over time",
        )
        base = Path(__file__).parent.parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base / "data" / "kalshi" / "markets")

    def run(self) -> AnalysisOutput:
        con = duckdb.connect()

        with self.progress("Computing per-ticker volume shares"):
            event_vol = con.execute(f"""
                WITH mkts AS (
                    SELECT ticker,
                           {CATEGORY_SQL} AS category
                    FROM '{self.markets_dir}/*.parquet'
                ),
                tv AS (
                    SELECT t.ticker, SUM(t.count) AS vol
                    FROM '{self.trades_dir}/*.parquet' t
                    GROUP BY t.ticker
                )
                SELECT tv.ticker,
                       COALESCE(m.category, 'unknown') AS category,
                       tv.vol                          AS ticker_vol,
                       tv.vol * 1.0 / SUM(tv.vol) OVER () AS share
                FROM tv
                LEFT JOIN mkts m ON tv.ticker = m.ticker
                ORDER BY tv.vol DESC
            """).df()

        event_vol["group"] = event_vol["category"].apply(get_group)

        total_vol = float(event_vol["ticker_vol"].sum())
        hhi = float((event_vol["share"] ** 2).sum())

        # Top-1% and top-10% share
        n = len(event_vol)
        top1_n  = max(1, int(np.ceil(n * 0.01)))
        top10_n = max(1, int(np.ceil(n * 0.10)))
        top1_share  = float(event_vol.head(top1_n)["share"].sum())
        top10_share = float(event_vol.head(top10_n)["share"].sum())

        with self.progress("HHI by quarterly cohort"):
            quarterly = con.execute(f"""
                WITH qt AS (
                    SELECT
                        DATE_TRUNC('quarter', t.created_time) AS quarter,
                        t.ticker,
                        SUM(t.count) AS vol
                    FROM '{self.trades_dir}/*.parquet' t
                    GROUP BY quarter, t.ticker
                ),
                qt_total AS (
                    SELECT quarter, SUM(vol) AS total_vol FROM qt GROUP BY quarter
                )
                SELECT
                    qt.quarter,
                    SUM((qt.vol * 1.0 / qt_total.total_vol) * (qt.vol * 1.0 / qt_total.total_vol)) AS hhi
                FROM qt
                INNER JOIN qt_total ON qt.quarter = qt_total.quarter
                GROUP BY qt.quarter
                ORDER BY qt.quarter
            """).df()

        quarterly["quarter"] = pd.to_datetime(quarterly["quarter"])

        summary = pd.DataFrame([
            {"metric": "global_hhi",     "value": round(hhi, 6)},
            {"metric": "n_tickers",      "value": n},
            {"metric": "top1_share",     "value": round(top1_share, 4)},
            {"metric": "top10_share",    "value": round(top10_share, 4)},
            {"metric": "total_volume",   "value": int(total_vol)},
        ])

        fig = self._make_figure(event_vol, quarterly, hhi, top1_share)
        return AnalysisOutput(figure=fig, data=summary)

    def _make_figure(
        self,
        event_vol: pd.DataFrame,
        quarterly: pd.DataFrame,
        hhi: float,
        top1_share: float,
    ) -> plt.Figure:
        from src.common.plot_style import (
            new_fig, clean_ax, stat_box, shade_v, BLUE, ORANGE, RED, AMBER, GRAY,
        )

        fig, axes = new_fig(1, 2, suptitle="Kalshi — Contract-Level Liquidity Concentration")

        # ── Left: HHI over time with benchmark bands ──────────────────────────
        ax = axes[0]
        # HHI interpretation zones (scaled 0–1; DOJ thresholds ÷ 10000)
        shade_v(ax, 0.15, 0.25, color=AMBER, alpha=0.12, label="Moderate (1500–2500)")
        shade_v(ax, 0.25, 1.00, color=RED,   alpha=0.07, label="Concentrated (>2500)")
        ax.plot(quarterly["quarter"], quarterly["hhi"],
                color=BLUE, lw=2.2, marker="o", ms=4.5, zorder=3,
                label="Quarterly HHI")
        ax.axhline(hhi, color=ORANGE, lw=1.4, linestyle=":", label=f"Global HHI = {hhi:.4f}")
        clean_ax(ax,
                 xlabel="Quarter",
                 ylabel="HHI (ticker-level, 0–1 scale)",
                 title="Volume Concentration Over Time",
                 zero_h=False)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8.5)

        # ── Right: Lorenz-style concentration curve ───────────────────────────
        ax2 = axes[1]
        sorted_shares = event_vol["share"].sort_values(ascending=False).values
        cumshare = np.cumsum(sorted_shares)
        n_tickers = len(sorted_shares)
        pct = np.arange(1, n_tickers + 1) / n_tickers * 100

        # Gini coefficient
        _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
        gini = 1 - 2 * _trapz(
            cumshare[::-1],
            np.linspace(0, 1, n_tickers),
        )

        ax2.plot(pct, cumshare * 100, color=ORANGE, lw=2.2, zorder=3)
        ax2.fill_between(pct, cumshare * 100, alpha=0.15, color=ORANGE, linewidth=0)
        ax2.plot([0, 100], [0, 100], "--", color=GRAY, lw=1.2, label="Perfect equality")
        ax2.axvline(1, color=RED, lw=1.3, linestyle="--",
                    label=f"Top 1% → {top1_share*100:.1f}% of volume")
        ax2.axvline(10, color=AMBER, lw=1.1, linestyle=":",
                    label=f"Top 10% → {event_vol.head(max(1, n_tickers // 10))['share'].sum()*100:.1f}% of volume")
        clean_ax(ax2,
                 xlabel="Percentile of Tickers (by volume, descending)",
                 ylabel="Cumulative Volume Share (%)",
                 title="Volume Concentration Curve (Lorenz)",
                 zero_h=False)
        ax2.legend(fontsize=8.5, loc="upper left")
        stat_box(ax2, f"Gini ≈ {gini:.3f}\nn = {n_tickers:,} tickers", loc="lower right")

        fig.tight_layout()
        return fig
