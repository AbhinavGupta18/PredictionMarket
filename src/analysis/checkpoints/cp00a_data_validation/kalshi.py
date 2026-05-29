"""CP00a — Data Validation: Kalshi dataset.

Validates schema consistency, date coverage, contract counts,
volume totals, null rates, and join quality.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd

from src.common.analysis import Analysis, AnalysisOutput


class KalshiDataValidation(Analysis):
    """Profile and validate the Kalshi trades + markets dataset."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="kalshi_cp00a_data_validation",
            description="Kalshi dataset validation: schema, coverage, nulls, quality flags",
        )
        base = Path(__file__).parent.parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base / "data" / "kalshi" / "markets")

    def run(self) -> AnalysisOutput:
        con = duckdb.connect()

        with self.progress("Profiling trades"):
            t = con.execute(f"""
                SELECT
                    COUNT(*)                                                    AS total_rows,
                    SUM(count)                                                  AS total_contracts,
                    COUNT(DISTINCT ticker)                                      AS unique_tickers,
                    MIN(created_time)                                           AS date_min,
                    MAX(created_time)                                           AS date_max,
                    COUNT(*) FILTER (WHERE taker_side IS NULL)                  AS null_taker_side,
                    COUNT(*) FILTER (WHERE yes_price  IS NULL)                  AS null_yes_price,
                    COUNT(*) FILTER (WHERE count       IS NULL)                 AS null_count,
                    COUNT(*) FILTER (WHERE ticker      IS NULL)                 AS null_ticker,
                    COUNT(*) FILTER (WHERE yes_price < 1 OR yes_price > 99)    AS out_of_range_price,
                    COUNT(*) FILTER (WHERE taker_side NOT IN ('yes','no'))      AS invalid_taker_side
                FROM '{self.trades_dir}/*.parquet'
            """).df().iloc[0]

        with self.progress("Profiling markets"):
            m = con.execute(f"""
                SELECT
                    COUNT(*)                                                        AS total_rows,
                    COUNT(DISTINCT ticker)                                          AS unique_tickers,
                    COUNT(DISTINCT event_ticker)                                    AS unique_events,
                    COUNT(*) FILTER (WHERE status = 'finalized')                    AS finalized,
                    COUNT(*) FILTER (WHERE status != 'finalized')                   AS not_finalized,
                    COUNT(*) FILTER (WHERE result = 'yes')                          AS result_yes,
                    COUNT(*) FILTER (WHERE result = 'no')                           AS result_no,
                    COUNT(*) FILTER (WHERE result NOT IN ('yes','no') OR result IS NULL) AS result_missing,
                    MIN(open_time)                                                  AS date_min,
                    MAX(close_time)                                                 AS date_max
                FROM '{self.markets_dir}/*.parquet'
            """).df().iloc[0]

        with self.progress("Checking join integrity"):
            orphan_tickers = con.execute(f"""
                SELECT COUNT(DISTINCT t.ticker)
                FROM '{self.trades_dir}/*.parquet'  t
                LEFT JOIN '{self.markets_dir}/*.parquet' m ON t.ticker = m.ticker
                WHERE m.ticker IS NULL
            """).fetchone()[0]

        with self.progress("Monthly volume profile"):
            monthly = con.execute(f"""
                SELECT
                    DATE_TRUNC('month', created_time) AS month,
                    COUNT(*)                          AS trade_rows,
                    SUM(count)                        AS contracts
                FROM '{self.trades_dir}/*.parquet'
                GROUP BY month ORDER BY month
            """).df()

        rows = [
            ("trades.total_rows",          int(t["total_rows"]),           "Trade-table row count"),
            ("trades.total_contracts",     int(t["total_contracts"]),      "Sum of all contract counts"),
            ("trades.unique_tickers",      int(t["unique_tickers"]),       "Distinct tickers in trades"),
            ("trades.date_min",            str(t["date_min"]),             "Earliest trade timestamp"),
            ("trades.date_max",            str(t["date_max"]),             "Latest trade timestamp"),
            ("trades.null_taker_side",     int(t["null_taker_side"]),      "Rows with NULL taker_side"),
            ("trades.null_yes_price",      int(t["null_yes_price"]),       "Rows with NULL yes_price"),
            ("trades.null_count",          int(t["null_count"]),           "Rows with NULL count"),
            ("trades.null_ticker",         int(t["null_ticker"]),          "Rows with NULL ticker"),
            ("trades.out_of_range_price",  int(t["out_of_range_price"]),   "Price outside [1, 99]"),
            ("trades.invalid_taker_side",  int(t["invalid_taker_side"]),   "taker_side not yes/no"),
            ("markets.total_rows",         int(m["total_rows"]),           "Markets-table row count"),
            ("markets.unique_tickers",     int(m["unique_tickers"]),       "Distinct tickers in markets"),
            ("markets.unique_events",      int(m["unique_events"]),        "Distinct event_ticker"),
            ("markets.finalized",          int(m["finalized"]),            "Status = finalized"),
            ("markets.not_finalized",      int(m["not_finalized"]),        "Status != finalized"),
            ("markets.result_yes",         int(m["result_yes"]),           "Markets resolved YES"),
            ("markets.result_no",          int(m["result_no"]),            "Markets resolved NO"),
            ("markets.result_missing",     int(m["result_missing"]),       "Missing/other result"),
            ("markets.date_min",           str(m["date_min"]),             "Earliest open_time"),
            ("markets.date_max",           str(m["date_max"]),             "Latest close_time"),
            ("join.orphan_tickers",        int(orphan_tickers),            "Trades with no market record"),
        ]
        summary = pd.DataFrame(rows, columns=["metric", "value", "description"])

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        axes[0].bar(monthly["month"], monthly["contracts"] / 1e6, color="#4C72B0", width=25)
        axes[0].set_title("Kalshi — Monthly Contract Volume")
        axes[0].set_ylabel("Contracts (millions)")
        axes[0].set_xlabel("Month")

        axes[1].bar(monthly["month"], monthly["trade_rows"] / 1e3, color="#DD8452", width=25)
        axes[1].set_title("Kalshi — Monthly Trade Row Count")
        axes[1].set_ylabel("Trade Rows (thousands)")
        axes[1].set_xlabel("Month")

        plt.tight_layout()
        return AnalysisOutput(figure=fig, data=summary)
