"""CP12 — Data Limitations: Polymarket.

Robustness checks and structural caveats for the Polymarket dataset:
block coverage gaps, unresolvable markets, FPMM vs CTF split, and
cross-platform comparability notes.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd

from src.common.analysis import Analysis, AnalysisOutput


class PolymarketDataLimitations(Analysis):
    """Polymarket data quality, coverage, and structural limitation report."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        legacy_trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
        blocks_dir: Path | str | None = None,
    ):
        super().__init__(
            name="polymarket_cp12_data_limitations",
            description="Polymarket: robustness checks, block gaps, market resolution caveats",
        )
        base = Path(__file__).parent.parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base / "data" / "polymarket" / "trades")
        self.legacy_dir = Path(legacy_trades_dir or base / "data" / "polymarket" / "legacy_trades")
        self.markets_dir = Path(markets_dir or base / "data" / "polymarket" / "markets")
        self.blocks_dir = Path(blocks_dir or base / "data" / "polymarket" / "blocks")

    def run(self) -> AnalysisOutput:
        con = duckdb.connect()

        with self.progress("Market resolution rate"):
            mkt = con.execute(f"""
                SELECT
                    COUNT(*)                                                AS total,
                    COUNT(*) FILTER (WHERE closed = true)                   AS closed,
                    COUNT(*) FILTER (WHERE outcome_prices IS NOT NULL
                                     AND closed = true)                     AS resolvable,
                    COUNT(*) FILTER (WHERE clob_token_ids IS NULL)          AS missing_tokens
                FROM '{self.markets_dir}/*.parquet'
            """).df().iloc[0]

        with self.progress("Block coverage gaps"):
            block_range = con.execute(f"""
                SELECT MIN(block_number) AS b_min, MAX(block_number) AS b_max,
                       COUNT(DISTINCT block_number) AS n_blocks
                FROM '{self.blocks_dir}/*.parquet'
            """).df().iloc[0]
            expected_blocks = int(block_range["b_max"]) - int(block_range["b_min"]) + 1
            missing_blocks = expected_blocks - int(block_range["n_blocks"])

        with self.progress("CTF vs legacy trade split"):
            ctf_n  = con.execute(f"SELECT COUNT(*) FROM '{self.trades_dir}/*.parquet'").fetchone()[0]
            leg_n  = con.execute(f"SELECT COUNT(*) FROM '{self.legacy_dir}/*.parquet'").fetchone()[0]

        rows = [
            ("markets.total",              int(mkt["total"]),      "Total markets in dataset"),
            ("markets.closed",             int(mkt["closed"]),     "Closed (resolved) markets"),
            ("markets.resolvable",         int(mkt["resolvable"]), "Closed with parseable outcome_prices"),
            ("markets.missing_tokens",     int(mkt["missing_tokens"]), "Markets with no CTF token IDs"),
            ("blocks.expected_range",      expected_blocks,        "Expected blocks in range [min, max]"),
            ("blocks.available",           int(block_range["n_blocks"]), "Blocks actually in dataset"),
            ("blocks.missing",             missing_blocks,         "Estimated missing block records"),
            ("trades.ctf_count",           int(ctf_n),             "CTF exchange trade rows"),
            ("trades.legacy_fpmm_count",   int(leg_n),             "Legacy FPMM trade rows"),
            ("trades.total",               int(ctf_n) + int(leg_n), "Total trade rows across both systems"),
        ]
        summary = pd.DataFrame(rows, columns=["metric", "value", "description"])

        limitations = pd.DataFrame([
            ("No maker/taker role",   "Polymarket CTF trades represent bilateral matches; taker identity is inferred from asset-flow direction only."),
            ("AMM vs CLOB",           "Legacy FPMM trades use automated market-maker pricing; CLOB CTF trades reflect limit-order matching. These are not directly comparable."),
            ("Token resolution",      "Only binary markets with 2-outcome token IDs and clear price resolution (>0.99 / <0.01) are included in calibration analyses."),
            ("On-chain latency",      "Block-level timestamps may differ from order submission time; intra-block ordering is not guaranteed."),
            ("Collateral diversity",  "Some markets used non-USDC collateral (USDC.e, etc.); these are excluded from absolute return calculations."),
        ], columns=["limitation", "description"])

        from src.common.plot_style import new_fig, clean_ax, BLUE, GREEN, ORANGE

        fig, axes = new_fig(1, 2, suptitle="Polymarket — Data Limitations & Coverage")

        vals   = [int(mkt["total"]), int(mkt["closed"]), int(mkt["resolvable"])]
        labels = ["All Markets", "Closed", "Resolvable"]
        clrs   = [BLUE, GREEN, ORANGE]
        axes[0].barh(labels[::-1], vals[::-1], color=clrs[::-1], alpha=0.88, linewidth=0)
        for i, v in enumerate(vals[::-1]):
            axes[0].text(v * 1.01, i, f"{v:,}", va="center", fontsize=8.5)
        clean_ax(axes[0], xlabel="Count",
                 title="Market Resolution Funnel", zero_h=False)

        axes[1].pie(
            [int(ctf_n), int(leg_n)],
            labels=[f"CTF\n{int(ctf_n):,}", f"Legacy FPMM\n{int(leg_n):,}"],
            colors=[BLUE, ORANGE],
            autopct="%1.1f%%", startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
            textprops={"fontsize": 9.5},
        )
        axes[1].set_title("Trade Volume: CTF vs Legacy FPMM", fontweight="semibold")

        fig.tight_layout()
        return AnalysisOutput(figure=fig, data=limitations)
