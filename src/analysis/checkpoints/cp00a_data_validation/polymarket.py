"""CP00a — Data Validation: Polymarket dataset.

Validates CTF trades, legacy FPMM trades, markets, and blocks tables.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd

from src.common.analysis import Analysis, AnalysisOutput


class PolymarketDataValidation(Analysis):
    """Profile and validate the Polymarket dataset."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        legacy_trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
        blocks_dir: Path | str | None = None,
    ):
        super().__init__(
            name="polymarket_cp00a_data_validation",
            description="Polymarket dataset validation: schema, coverage, nulls, quality flags",
        )
        base = Path(__file__).parent.parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base / "data" / "polymarket" / "trades")
        self.legacy_dir = Path(legacy_trades_dir or base / "data" / "polymarket" / "legacy_trades")
        self.markets_dir = Path(markets_dir or base / "data" / "polymarket" / "markets")
        self.blocks_dir = Path(blocks_dir or base / "data" / "polymarket" / "blocks")

    def run(self) -> AnalysisOutput:
        con = duckdb.connect()

        with self.progress("Profiling CTF trades"):
            ctf = con.execute(f"""
                SELECT
                    COUNT(*)                                            AS total_rows,
                    COUNT(DISTINCT block_number)                        AS unique_blocks,
                    MIN(block_number)                                   AS block_min,
                    MAX(block_number)                                   AS block_max,
                    COUNT(*) FILTER (WHERE maker_amount IS NULL)        AS null_maker_amount,
                    COUNT(*) FILTER (WHERE taker_amount IS NULL)        AS null_taker_amount,
                    COUNT(*) FILTER (WHERE maker_amount <= 0 OR taker_amount <= 0) AS zero_amount_rows
                FROM '{self.trades_dir}/*.parquet'
            """).df().iloc[0]

        with self.progress("Profiling legacy FPMM trades"):
            leg = con.execute(f"""
                SELECT
                    COUNT(*)                                            AS total_rows,
                    COUNT(DISTINCT block_number)                        AS unique_blocks,
                    MIN(block_number)                                   AS block_min,
                    MAX(block_number)                                   AS block_max,
                    COUNT(*) FILTER (WHERE amount IS NULL)              AS null_amount,
                    COUNT(*) FILTER (WHERE outcome_tokens IS NULL)      AS null_tokens
                FROM '{self.legacy_dir}/*.parquet'
            """).df().iloc[0]

        with self.progress("Profiling markets"):
            mkts = con.execute(f"""
                SELECT
                    COUNT(*)                                                AS total_rows,
                    COUNT(*) FILTER (WHERE closed = true)                   AS closed_markets,
                    COUNT(*) FILTER (WHERE closed = false OR closed IS NULL) AS open_markets,
                    COUNT(*) FILTER (WHERE clob_token_ids IS NULL)          AS null_token_ids,
                    COUNT(*) FILTER (WHERE outcome_prices IS NULL)          AS null_prices
                FROM '{self.markets_dir}/*.parquet'
            """).df().iloc[0]

        with self.progress("Profiling blocks"):
            blks = con.execute(f"""
                SELECT
                    COUNT(*)             AS total_rows,
                    MIN(block_number)    AS block_min,
                    MAX(block_number)    AS block_max,
                    MIN(timestamp)       AS date_min,
                    MAX(timestamp)       AS date_max
                FROM '{self.blocks_dir}/*.parquet'
            """).df().iloc[0]

        with self.progress("Monthly block distribution"):
            monthly = con.execute(f"""
                SELECT
                    DATE_TRUNC('month', timestamp::TIMESTAMP) AS month,
                    COUNT(*)                                  AS blocks
                FROM '{self.blocks_dir}/*.parquet'
                GROUP BY month ORDER BY month
            """).df()

        rows = [
            ("ctf_trades.total_rows",       int(ctf["total_rows"]),         "CTF trade rows"),
            ("ctf_trades.unique_blocks",    int(ctf["unique_blocks"]),      "Unique blocks in CTF trades"),
            ("ctf_trades.block_min",        int(ctf["block_min"]),          "Earliest CTF block"),
            ("ctf_trades.block_max",        int(ctf["block_max"]),          "Latest CTF block"),
            ("ctf_trades.null_amounts",     int(ctf["null_maker_amount"]),  "Rows with NULL maker_amount"),
            ("ctf_trades.zero_amounts",     int(ctf["zero_amount_rows"]),   "Rows with zero amounts"),
            ("legacy_trades.total_rows",    int(leg["total_rows"]),         "Legacy FPMM trade rows"),
            ("legacy_trades.unique_blocks", int(leg["unique_blocks"]),      "Unique blocks in legacy trades"),
            ("legacy_trades.null_amount",   int(leg["null_amount"]),        "Rows with NULL amount"),
            ("markets.total_rows",          int(mkts["total_rows"]),        "Market rows"),
            ("markets.closed",              int(mkts["closed_markets"]),    "Closed (resolved) markets"),
            ("markets.open",                int(mkts["open_markets"]),      "Open/active markets"),
            ("markets.null_token_ids",      int(mkts["null_token_ids"]),    "Markets with NULL token IDs"),
            ("markets.null_prices",         int(mkts["null_prices"]),       "Markets with NULL prices"),
            ("blocks.total_rows",           int(blks["total_rows"]),        "Block rows"),
            ("blocks.date_min",             str(blks["date_min"]),          "Earliest block timestamp"),
            ("blocks.date_max",             str(blks["date_max"]),          "Latest block timestamp"),
        ]
        summary = pd.DataFrame(rows, columns=["metric", "value", "description"])

        from src.common.plot_style import new_fig, clean_ax, BLUE

        fig, ax = new_fig(1, 1, figsize=(13, 4.5))
        ax.bar(monthly["month"], monthly["blocks"], width=25,
               color=BLUE, alpha=0.80, linewidth=0)
        clean_ax(ax,
                 xlabel="Month",
                 ylabel="Blocks",
                 title="Polymarket — Monthly Block Count (Dataset Coverage)",
                 zero_h=False)
        fig.tight_layout()

        return AnalysisOutput(figure=fig, data=summary)
