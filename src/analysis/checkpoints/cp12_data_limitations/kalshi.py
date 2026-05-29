"""CP12 — Institutional Limitations & Data Caveats: Kalshi.

Documents structural constraints, runs robustness and data-quality checks,
quantifies potential biases (survival bias, position-limit censoring), and
produces a validation report confirming which analyses use clean data.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd

from src.common.analysis import Analysis, AnalysisOutput


class KalshiDataLimitations(Analysis):
    """Data quality, robustness checks, and structural limitation report."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="kalshi_cp12_data_limitations",
            description="Kalshi: robustness checks, survival bias audit, structural caveats",
        )
        base = Path(__file__).parent.parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base / "data" / "kalshi" / "markets")

    def run(self) -> AnalysisOutput:
        con = duckdb.connect()

        # --- 1. Survival bias: fraction of traded markets that are finalized ---
        with self.progress("Survival bias check"):
            sb = con.execute(f"""
                SELECT
                    COUNT(DISTINCT t.ticker)                                        AS traded_tickers,
                    COUNT(DISTINCT m.ticker) FILTER (WHERE m.status = 'finalized') AS finalized_tickers,
                    COUNT(DISTINCT m.ticker) FILTER (WHERE m.status != 'finalized') AS non_finalized_tickers
                FROM '{self.trades_dir}/*.parquet' t
                LEFT JOIN '{self.markets_dir}/*.parquet' m ON t.ticker = m.ticker
            """).df().iloc[0]

        # --- 2. Outlier censoring: trade count (position size) distribution ---
        with self.progress("Contract size distribution"):
            size_dist = con.execute(f"""
                SELECT
                    percentile_cont(0.50) WITHIN GROUP (ORDER BY count) AS p50,
                    percentile_cont(0.90) WITHIN GROUP (ORDER BY count) AS p90,
                    percentile_cont(0.95) WITHIN GROUP (ORDER BY count) AS p95,
                    percentile_cont(0.99) WITHIN GROUP (ORDER BY count) AS p99,
                    MAX(count)                                          AS max_count,
                    AVG(count)                                          AS mean_count,
                    COUNT(*) FILTER (WHERE count > 1000)               AS large_trades
                FROM '{self.trades_dir}/*.parquet'
            """).df().iloc[0]

        # --- 3. Temporal gap check: days with no trades ---
        with self.progress("Daily coverage"):
            daily = con.execute(f"""
                SELECT
                    DATE_TRUNC('day', created_time) AS day,
                    COUNT(*) AS n_trades
                FROM '{self.trades_dir}/*.parquet'
                GROUP BY day
                ORDER BY day
            """).df()
        daily["day"] = pd.to_datetime(daily["day"])
        date_range = pd.date_range(daily["day"].min(), daily["day"].max(), freq="D")
        missing_days = len(date_range) - len(daily)

        # --- 4. Price edge cases ---
        with self.progress("Edge price checks"):
            edge = con.execute(f"""
                SELECT
                    COUNT(*) FILTER (WHERE yes_price = 1)  AS price_at_1,
                    COUNT(*) FILTER (WHERE yes_price = 99) AS price_at_99,
                    COUNT(*) FILTER (WHERE yes_price = 50) AS price_at_50
                FROM '{self.trades_dir}/*.parquet'
            """).df().iloc[0]

        rows = [
            ("survival.traded_tickers",         int(sb["traded_tickers"]),         "Tickers with at least 1 trade"),
            ("survival.finalized_tickers",       int(sb["finalized_tickers"]),      "Traded tickers that are finalized"),
            ("survival.non_finalized_tickers",   int(sb["non_finalized_tickers"]),  "Traded tickers still open (excluded from PnL)"),
            ("survival.finalized_rate_pct",
             round(int(sb["finalized_tickers"]) / max(int(sb["traded_tickers"]), 1) * 100, 2),
             "Fraction of traded tickers with final result"),
            ("size.p50_contracts",               float(size_dist["p50"]),           "Median trade size (contracts)"),
            ("size.p90_contracts",               float(size_dist["p90"]),           "90th-pct trade size"),
            ("size.p99_contracts",               float(size_dist["p99"]),           "99th-pct trade size"),
            ("size.max_contracts",               float(size_dist["max_count"]),     "Largest single trade"),
            ("size.large_trades_gt1000",         int(size_dist["large_trades"]),    "Trades > 1000 contracts (potential outliers)"),
            ("temporal.missing_days",            missing_days,                      "Calendar days with zero trades"),
            ("edge.price_at_1cent",              int(edge["price_at_1"]),           "Trades at minimum price (1¢)"),
            ("edge.price_at_99cent",             int(edge["price_at_99"]),          "Trades at maximum price (99¢)"),
            ("edge.price_at_50cent",             int(edge["price_at_50"]),          "Trades at 50¢ (max uncertainty)"),
        ]
        summary = pd.DataFrame(rows, columns=["metric", "value", "description"])

        # Structural limitations table (qualitative)
        limitations = pd.DataFrame([
            ("Position limits",     "Kalshi caps individual positions; large informed bets may be censored, biasing Maker returns upward."),
            ("No maker identity",   "Public trade data lacks maker IDs; HHI proxied by event-level volume, not true LP concentration."),
            ("Survival bias",       f"{int(sb['non_finalized_tickers'])} traded markets excluded from PnL analyses (not yet finalized)."),
            ("Platform fragmentation", "No cross-platform arbitrage paths modelled; liquidity conditions on Polymarket not controlled for."),
            ("Fee structure",       "Kalshi charges taker fees that are not reflected in raw yes_price; returns are gross of fees."),
            ("Data coverage",       "Dataset covers public markets; internal/restricted markets are not included."),
        ], columns=["limitation", "description"])

        fig = self._make_figure(daily, summary)
        return AnalysisOutput(figure=fig, data=limitations)

    def _make_figure(self, daily: pd.DataFrame, summary: pd.DataFrame) -> plt.Figure:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Daily trade activity (gap detection)
        axes[0].bar(daily["day"], daily["n_trades"], width=1, color="#4C72B0", alpha=0.7)
        axes[0].set_xlabel("Date")
        axes[0].set_ylabel("Trades per Day")
        axes[0].set_title("Kalshi Daily Trade Activity (Coverage Check)")

        # Summary metrics as horizontal text table
        axes[1].axis("off")
        table_data = [[row["metric"], str(row["value"]), row["description"]]
                      for _, row in summary.iterrows()]
        tbl = axes[1].table(
            cellText=table_data,
            colLabels=["Metric", "Value", "Description"],
            loc="center",
            cellLoc="left",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.auto_set_column_width([0, 1, 2])
        axes[1].set_title("Data Quality Metrics", pad=10)

        plt.tight_layout()
        return fig
