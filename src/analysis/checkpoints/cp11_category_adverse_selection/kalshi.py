"""CP11 — Adverse Selection & Participant Heterogeneity: Kalshi.

Measures price-impact velocity (|ΔP| per trade) in the periods immediately
following a Taker trade, stratified by category group.

Hypothesis:
  Finance / Macro categories → faster price reversal (informed takers → adverse selection)
  Sports / Entertainment    → slower price decay (noise/sentiment-driven takers)
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.kalshi.util.categories import CATEGORY_SQL, get_group
from src.common.analysis import Analysis, AnalysisOutput


class KalshiCategoryAdverseSelection(Analysis):
    """Post-trade price velocity as an adverse selection proxy across categories."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="kalshi_cp11_category_adverse_selection",
            description="Price velocity post-taker trade by category: adverse selection test",
        )
        base = Path(__file__).parent.parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base / "data" / "kalshi" / "markets")

    def run(self) -> AnalysisOutput:
        con = duckdb.connect()

        with self.progress("Computing trade-to-trade price changes with category"):
            df = con.execute(f"""
                WITH mkts AS (
                    SELECT ticker,
                           {CATEGORY_SQL} AS category
                    FROM '{self.markets_dir}/*.parquet'
                ),
                ordered AS (
                    SELECT
                        t.ticker,
                        m.category,
                        t.yes_price,
                        t.created_time,
                        LAG(t.yes_price) OVER (
                            PARTITION BY t.ticker ORDER BY t.created_time
                        ) AS prev_price
                    FROM '{self.trades_dir}/*.parquet' t
                    INNER JOIN mkts m ON t.ticker = m.ticker
                )
                SELECT
                    category,
                    ABS(yes_price - prev_price)     AS abs_price_change,
                    yes_price - prev_price           AS price_change
                FROM ordered
                WHERE prev_price IS NOT NULL
            """).df()

        df["group"] = df["category"].apply(get_group)

        # Per-group: mean |ΔP|, median |ΔP|, autocorrelation of signed ΔP
        group_stats = []
        for grp, gdf in df.groupby("group"):
            pc = gdf["price_change"].dropna()
            apc = gdf["abs_price_change"].dropna()
            # First-order autocorrelation as a reversal proxy
            if len(pc) > 10:
                ac1 = float(pc.autocorr(lag=1))
            else:
                ac1 = float("nan")
            group_stats.append({
                "group":             grp,
                "n_price_changes":   len(apc),
                "mean_abs_change":   float(apc.mean()),
                "median_abs_change": float(apc.median()),
                "std_abs_change":    float(apc.std()),
                "autocorr_lag1":     ac1,
            })

        gdf_stats = pd.DataFrame(group_stats).sort_values("mean_abs_change", ascending=False)

        # Finance vs Sports t-test
        finance_grps = {"Finance", "Crypto"}
        sports_grps  = {"Sports", "Entertainment"}
        fin_vals = df.loc[df["group"].isin(finance_grps), "abs_price_change"].dropna()
        spt_vals = df.loc[df["group"].isin(sports_grps),  "abs_price_change"].dropna()
        if len(fin_vals) > 10 and len(spt_vals) > 10:
            t_stat, p_val = stats.ttest_ind(fin_vals, spt_vals, equal_var=False)
        else:
            t_stat, p_val = float("nan"), float("nan")

        gdf_stats.loc[len(gdf_stats)] = {
            "group": "__Finance_vs_Sports__",
            "n_price_changes": 0,
            "mean_abs_change": float("nan"),
            "median_abs_change": float("nan"),
            "std_abs_change": float("nan"),
            "autocorr_lag1": float("nan"),
        }
        gdf_stats.attrs["finance_vs_sports_t"] = float(t_stat)
        gdf_stats.attrs["finance_vs_sports_p"] = float(p_val)

        fig = self._make_figure(gdf_stats.iloc[:-1], float(t_stat), float(p_val))
        return AnalysisOutput(figure=fig, data=gdf_stats)

    def _make_figure(
        self,
        gdf: pd.DataFrame,
        t_stat: float,
        p_val: float,
    ) -> plt.Figure:
        from src.common.plot_style import (
            new_fig, clean_ax, stat_box, sig_stars, bar_colors, group_color,
            BLUE, GREEN, RED, GRAY,
        )

        fig, axes = new_fig(1, 2, suptitle="Kalshi — Adverse Selection by Category")
        stars = sig_stars(p_val)
        top = gdf.head(10)

        # ── Left: mean |ΔP| (price impact velocity) ──────────────────────────
        ax = axes[0]
        grp_list = list(reversed(top["group"].tolist()))
        vel_list = list(reversed(top["mean_abs_change"].tolist()))
        clrs = [group_color(g) for g in grp_list]
        ax.barh(grp_list, vel_list, color=clrs, alpha=0.88, linewidth=0)
        clean_ax(ax,
                 xlabel="Mean |ΔP| (cents)",
                 title="Post-Trade Price Velocity by Category",
                 zero_h=False)
        stat_box(ax,
                 f"Finance vs Sports\nt = {t_stat:.2f},  p = {p_val:.4f} {stars}")

        # ── Right: lag-1 autocorrelation of ΔP ───────────────────────────────
        ax2 = axes[1]
        valid = gdf.dropna(subset=["autocorr_lag1"]).head(10)
        grp2 = list(reversed(valid["group"].tolist()))
        ac   = list(reversed(valid["autocorr_lag1"].tolist()))
        clrs2 = [RED if a < 0 else GREEN for a in ac]
        ax2.barh(grp2, ac, color=clrs2, alpha=0.88, linewidth=0)
        clean_ax(ax2,
                 xlabel="Lag-1 Autocorrelation of ΔP",
                 title="Price-Change Autocorrelation\n(negative → reversal → informed flow)",
                 zero_h=False, zero_v=True)

        fig.tight_layout()
        return fig
