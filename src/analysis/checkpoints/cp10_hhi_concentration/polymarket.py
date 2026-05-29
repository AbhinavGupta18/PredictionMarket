"""CP10 — Professionalization of Liquidity: Polymarket.

Computes Herfindahl-Hirschman Index (HHI) for token-level volume concentration
on Polymarket's CTF exchange, with a quarterly cohort trend.

Note: Polymarket has no named categories. We use clob_token_ids to identify
individual binary outcome tokens and measure how concentrated CTF trading is
across tokens over time.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.common.analysis import Analysis, AnalysisOutput


class PolymarketHHIConcentration(Analysis):
    """HHI and top-1% volume concentration of Polymarket CTF token trading."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
        blocks_dir: Path | str | None = None,
    ):
        super().__init__(
            name="polymarket_cp10_hhi_concentration",
            description="HHI token-level volume concentration + top-1% share over time",
        )
        base = Path(__file__).parent.parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base / "data" / "polymarket" / "trades")
        self.markets_dir = Path(markets_dir or base / "data" / "polymarket" / "markets")
        self.blocks_dir = Path(blocks_dir or base / "data" / "polymarket" / "blocks")

    def run(self) -> AnalysisOutput:
        con = duckdb.connect()

        # ── Token-level volume shares (global) ────────────────────────────────
        with self.progress("Computing per-token volume shares"):
            token_vol = con.execute(f"""
                WITH tv AS (
                    SELECT
                        CASE WHEN maker_asset_id = '0' THEN taker_asset_id
                             ELSE maker_asset_id END AS token_id,
                        SUM(taker_amount) AS vol
                    FROM '{self.trades_dir}/*.parquet'
                    WHERE taker_amount > 0
                    GROUP BY token_id
                )
                SELECT token_id,
                       vol,
                       vol * 1.0 / SUM(vol) OVER () AS share
                FROM tv
                WHERE token_id IS NOT NULL
                ORDER BY vol DESC
            """).df()

        hhi = float((token_vol["share"] ** 2).sum())
        n = len(token_vol)
        top1_n  = max(1, int(np.ceil(n * 0.01)))
        top10_n = max(1, int(np.ceil(n * 0.10)))
        top1_share  = float(token_vol.head(top1_n)["share"].sum())
        top10_share = float(token_vol.head(top10_n)["share"].sum())

        # ── Quarterly HHI trend ───────────────────────────────────────────────
        with self.progress("HHI by quarterly cohort"):
            quarterly = con.execute(f"""
                WITH qt AS (
                    SELECT
                        DATE_TRUNC('quarter', b.timestamp::TIMESTAMP) AS quarter,
                        CASE WHEN t.maker_asset_id = '0' THEN t.taker_asset_id
                             ELSE t.maker_asset_id END                AS token_id,
                        SUM(t.taker_amount) AS vol
                    FROM '{self.trades_dir}/*.parquet' t
                    JOIN '{self.blocks_dir}/*.parquet' b ON t.block_number = b.block_number
                    WHERE t.taker_amount > 0
                    GROUP BY quarter, token_id
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
            {"metric": "n_tokens",       "value": n},
            {"metric": "top1_share",     "value": round(top1_share, 4)},
            {"metric": "top10_share",    "value": round(top10_share, 4)},
            {"metric": "total_volume",   "value": int(token_vol["vol"].sum())},
        ])

        fig = self._make_figure(token_vol, quarterly, hhi, top1_share, n, top10_n, top10_share)
        return AnalysisOutput(figure=fig, data=summary)

    def _make_figure(
        self,
        token_vol: pd.DataFrame,
        quarterly: pd.DataFrame,
        hhi: float,
        top1_share: float,
        n: int,
        top10_n: int,
        top10_share: float,
    ) -> "plt.Figure":
        import matplotlib.pyplot as plt
        from src.common.plot_style import (
            new_fig, clean_ax, stat_box, shade_v, BLUE, ORANGE, RED, AMBER, GRAY,
        )

        fig, axes = new_fig(1, 2, suptitle="Polymarket — Token-Level Liquidity Concentration")

        # ── Left: HHI over time ───────────────────────────────────────────────
        ax = axes[0]
        shade_v(ax, 0.15, 0.25, color=AMBER, alpha=0.12)
        shade_v(ax, 0.25, 1.00, color=RED,   alpha=0.07)
        ax.plot(quarterly["quarter"], quarterly["hhi"],
                color=BLUE, lw=2.2, marker="o", ms=4.5, zorder=3,
                label="Quarterly HHI")
        ax.axhline(hhi, color=ORANGE, lw=1.4, linestyle=":",
                   label=f"Global HHI = {hhi:.4f}")
        clean_ax(ax,
                 xlabel="Quarter",
                 ylabel="HHI (token-level, 0–1 scale)",
                 title="Volume Concentration Over Time",
                 zero_h=False)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8.5)

        # ── Right: Lorenz-style concentration curve ───────────────────────────
        ax2 = axes[1]
        sorted_shares = token_vol["share"].sort_values(ascending=False).values
        cumshare = np.cumsum(sorted_shares)
        pct = np.arange(1, n + 1) / n * 100

        _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
        gini = 1 - 2 * _trapz(
            cumshare[::-1],
            np.linspace(0, 1, n),
        )

        ax2.plot(pct, cumshare * 100, color=ORANGE, lw=2.2, zorder=3)
        ax2.fill_between(pct, cumshare * 100, alpha=0.15, color=ORANGE, linewidth=0)
        ax2.plot([0, 100], [0, 100], "--", color=GRAY, lw=1.2, label="Perfect equality")
        ax2.axvline(1, color=RED, lw=1.3, linestyle="--",
                    label=f"Top 1% → {top1_share*100:.1f}% of volume")
        ax2.axvline(10, color=AMBER, lw=1.1, linestyle=":",
                    label=f"Top 10% → {top10_share*100:.1f}% of volume")
        clean_ax(ax2,
                 xlabel="Percentile of Tokens (by volume, descending)",
                 ylabel="Cumulative Volume Share (%)",
                 title="Volume Concentration Curve (Lorenz)",
                 zero_h=False)
        ax2.legend(fontsize=8.5, loc="upper left")
        stat_box(ax2, f"Gini ≈ {gini:.3f}\nn = {n:,} tokens", loc="lower right")

        fig.tight_layout()
        return fig
