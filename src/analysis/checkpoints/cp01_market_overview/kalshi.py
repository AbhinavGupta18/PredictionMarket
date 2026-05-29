"""CP01 — Market Architecture & Contract Design: Kalshi.

Extracts finalized markets, computes contract duration, stratifies by
category group, and generates a summary statistics table.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.kalshi.util.categories import CATEGORY_SQL, get_group
from src.common.analysis import Analysis, AnalysisOutput


class KalshiMarketOverview(Analysis):
    """Market architecture: contract duration and category distribution."""

    def __init__(self, markets_dir: Path | str | None = None):
        super().__init__(
            name="kalshi_cp01_market_overview",
            description="Contract design: duration, category breakdown, summary stats",
        )
        base = Path(__file__).parent.parent.parent.parent.parent
        self.markets_dir = Path(markets_dir or base / "data" / "kalshi" / "markets")

    def run(self) -> AnalysisOutput:
        con = duckdb.connect()

        with self.progress("Loading finalized markets"):
            df = con.execute(f"""
                SELECT
                    ticker,
                    event_ticker,
                    status,
                    result,
                    open_time,
                    close_time,
                    DATEDIFF('hour', open_time::TIMESTAMP, close_time::TIMESTAMP) AS duration_hours
                FROM '{self.markets_dir}/*.parquet'
                WHERE status = 'finalized'
                  AND result IN ('yes', 'no')
                  AND open_time IS NOT NULL
                  AND close_time IS NOT NULL
            """).df()

        df["group"] = df["event_ticker"].apply(
            lambda x: get_group(x) if isinstance(x, str) else "Other"
        )

        # --- Summary stats by group ---
        group_stats = (
            df.groupby("group")
            .agg(
                n_contracts=("ticker", "count"),
                pct_yes=("result", lambda s: 100.0 * (s == "yes").sum() / len(s)),
                median_duration_h=("duration_hours", "median"),
                mean_duration_h=("duration_hours", "mean"),
                p10_duration_h=("duration_hours", lambda s: np.percentile(s.dropna(), 10)),
                p90_duration_h=("duration_hours", lambda s: np.percentile(s.dropna(), 90)),
            )
            .reset_index()
            .sort_values("n_contracts", ascending=False)
        )

        # --- Global summary ---
        overall = pd.DataFrame([{
            "group": "ALL",
            "n_contracts":      len(df),
            "pct_yes":          100.0 * (df["result"] == "yes").mean(),
            "median_duration_h": df["duration_hours"].median(),
            "mean_duration_h":   df["duration_hours"].mean(),
            "p10_duration_h":    np.percentile(df["duration_hours"].dropna(), 10),
            "p90_duration_h":    np.percentile(df["duration_hours"].dropna(), 90),
        }])
        summary = pd.concat([overall, group_stats], ignore_index=True)

        # --- Figure ---
        from src.common.plot_style import new_fig, clean_ax, group_color, BLUE, ORANGE

        fig, axes = new_fig(1, 2, suptitle="Kalshi — Market Architecture Summary")
        top = group_stats.head(10)
        grp_colors = [group_color(g) for g in reversed(top["group"].tolist())]

        axes[0].barh(top["group"][::-1], top["n_contracts"][::-1],
                     color=grp_colors, alpha=0.88, linewidth=0)
        clean_ax(axes[0], xlabel="Number of Finalized Contracts",
                 title="Contract Count by Category", zero_h=False)

        axes[1].barh(top["group"][::-1], top["median_duration_h"][::-1],
                     color=grp_colors, alpha=0.88, linewidth=0)
        clean_ax(axes[1], xlabel="Median Duration (hours)",
                 title="Median Contract Duration by Category", zero_h=False)

        fig.tight_layout()
        return AnalysisOutput(figure=fig, data=summary)
