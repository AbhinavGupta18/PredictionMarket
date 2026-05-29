"""CP01 — Market Architecture & Contract Design: Polymarket.

Extracts closed/resolved markets, profiles resolution rates, volume over
time via block timestamps, and contract-duration proxies using block gaps.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.common.analysis import Analysis, AnalysisOutput


class PolymarketMarketOverview(Analysis):
    """Market architecture on Polymarket: resolution rates and activity over time."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
        blocks_dir: Path | str | None = None,
    ):
        super().__init__(
            name="polymarket_cp01_market_overview",
            description="Polymarket market structure: resolution rates, volume timeline, binary outcomes",
        )
        base = Path(__file__).parent.parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base / "data" / "polymarket" / "trades")
        self.markets_dir = Path(markets_dir or base / "data" / "polymarket" / "markets")
        self.blocks_dir = Path(blocks_dir or base / "data" / "polymarket" / "blocks")

    def run(self) -> AnalysisOutput:
        con = duckdb.connect()

        # ── Market resolution profile ─────────────────────────────────────────
        with self.progress("Loading market metadata"):
            mkts = con.execute(f"""
                SELECT
                    id,
                    closed,
                    clob_token_ids,
                    outcome_prices
                FROM '{self.markets_dir}/*.parquet'
            """).df()

        total_markets = len(mkts)
        closed_markets = int(mkts["closed"].sum())
        open_markets = total_markets - closed_markets

        # Outcome parsing: count resolvable (binary, clear winner)
        resolvable = 0
        outcome_yes_count = 0
        for _, row in mkts.iterrows():
            if not row["closed"]:
                continue
            try:
                prices = json.loads(row["outcome_prices"] or "[]")
                tokens = json.loads(row["clob_token_ids"] or "[]")
                if len(prices) == 2 and len(tokens) == 2:
                    p0, p1 = float(prices[0]), float(prices[1])
                    if (p0 > 0.99 and p1 < 0.01) or (p0 < 0.01 and p1 > 0.99):
                        resolvable += 1
                        if p0 > 0.99:
                            outcome_yes_count += 1
            except (json.JSONDecodeError, ValueError, TypeError):
                continue

        pct_yes = outcome_yes_count / resolvable * 100 if resolvable else float("nan")

        # ── Monthly trade volume via block timestamps ─────────────────────────
        with self.progress("Monthly CTF trade volume"):
            monthly = con.execute(f"""
                SELECT
                    DATE_TRUNC('month', b.timestamp::TIMESTAMP) AS month,
                    COUNT(*)                                     AS n_trades,
                    COUNT(DISTINCT t.maker_asset_id)             AS unique_tokens
                FROM '{self.trades_dir}/*.parquet' t
                JOIN '{self.blocks_dir}/*.parquet' b ON t.block_number = b.block_number
                GROUP BY month
                ORDER BY month
            """).df()
            monthly["month"] = pd.to_datetime(monthly["month"])

        # ── Resolution rate over time (by market creation quarter) ────────────
        # Proxy: proportion of closed markets with resolvable outcomes — global only
        resolution_rate = resolvable / closed_markets * 100 if closed_markets else float("nan")

        summary = pd.DataFrame([
            {"metric": "total_markets",       "value": total_markets},
            {"metric": "closed_markets",      "value": closed_markets},
            {"metric": "open_markets",        "value": open_markets},
            {"metric": "resolvable_markets",  "value": resolvable},
            {"metric": "resolution_rate_pct", "value": round(resolution_rate, 2)},
            {"metric": "pct_yes_outcomes",    "value": round(pct_yes, 2)},
            {"metric": "total_ctf_trades",    "value": int(monthly["n_trades"].sum())},
        ])

        fig = self._make_figure(monthly, total_markets, closed_markets,
                                resolvable, resolution_rate, pct_yes)
        return AnalysisOutput(figure=fig, data=summary)

    def _make_figure(
        self,
        monthly: pd.DataFrame,
        total: int,
        closed: int,
        resolvable: int,
        res_rate: float,
        pct_yes: float,
    ) -> "plt.Figure":
        import matplotlib.pyplot as plt
        from src.common.plot_style import (
            new_fig, clean_ax, stat_box, BLUE, GREEN, ORANGE, GRAY,
        )

        fig, axes = new_fig(1, 2, suptitle="Polymarket — Market Architecture Overview")

        # ── Left: monthly trade volume (bar chart) ───────────────────────────
        ax = axes[0]
        ax.bar(monthly["month"], monthly["n_trades"] / 1_000,
               width=25, color=BLUE, alpha=0.80, linewidth=0, label="CTF Trades")
        clean_ax(ax,
                 xlabel="Month",
                 ylabel="Trades (thousands)",
                 title="Monthly CTF Trade Volume",
                 zero_h=False)
        stat_box(ax, f"Total trades: {int(monthly['n_trades'].sum()):,}", loc="upper left")

        # ── Right: resolution funnel (horizontal bars) ───────────────────────
        ax2 = axes[1]
        labels = ["All Markets", "Closed", "Resolvable (binary)"]
        vals   = [total, closed, resolvable]
        clrs   = [BLUE, GREEN, ORANGE]
        ax2.barh(labels[::-1], vals[::-1], color=clrs[::-1], alpha=0.88, linewidth=0)
        for i, v in enumerate(vals[::-1]):
            ax2.text(v * 1.01, i, f"{v:,}", va="center", fontsize=8.5)
        clean_ax(ax2,
                 xlabel="Count",
                 title="Market Resolution Funnel",
                 zero_h=False)
        stat_box(ax2,
                 f"Resolution rate: {res_rate:.1f}%\nYES outcomes: {pct_yes:.1f}%",
                 loc="lower right")

        fig.tight_layout()
        return fig
