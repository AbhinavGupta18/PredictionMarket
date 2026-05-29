"""CP06 — Temporal Dynamics of Market Efficiency: Kalshi.

Three complementary temporal views:
  1. Maker-Taker Returns Over Time — quarterly VWR for both sides
  2. Taker Volume Distribution by Price — stacked-area composition over quarters
  3. Monthly OLS trend — regression of monthly maker VWR on time index
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import duckdb
from scipy import stats

from src.common.analysis import Analysis, AnalysisOutput
from src.common.metrics import MAKER_PNL_SQL, TAKER_PNL_SQL

BUCKET_LABELS = ["1–10", "11–20", "21–30", "31–40", "41–50",
                 "51–60", "61–70", "71–80", "81–90", "91–99"]
BUCKET_COLORS = ["#4361EE", "#06D6A0", "#F9C74F", "#EF233C", "#8338EC",
                 "#FB5607", "#48CAE4", "#84CC16", "#9B5DE5", "#F15BB5"]


class KalshiTemporalDynamics(Analysis):
    """3-panel temporal analysis: quarterly dual-returns, stacked-area composition, OLS trend."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="kalshi_cp06_temporal_dynamics",
            description="Temporal maker-alpha: quarterly dual returns, stacked-area, OLS trend",
        )
        base = Path(__file__).parent.parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base / "data" / "kalshi" / "markets")

    def run(self) -> AnalysisOutput:
        con = duckdb.connect()

        # ── 1. Monthly maker VWR for OLS trend ───────────────────────────────
        with self.progress("Monthly maker VWR for OLS"):
            monthly_raw = con.execute(f"""
                WITH resolved AS (
                    SELECT ticker, result
                    FROM '{self.markets_dir}/*.parquet'
                    WHERE status = 'finalized' AND result IN ('yes','no')
                )
                SELECT
                    DATE_TRUNC('month', t.created_time) AS month,
                    {MAKER_PNL_SQL}                     AS maker_pnl,
                    t.count                             AS contracts
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN resolved m ON t.ticker = m.ticker
            """).df()

        monthly = (
            monthly_raw.groupby("month")
            .apply(lambda g: pd.Series({
                "n_contracts": g["contracts"].sum(),
                "maker_vwr":   (g["maker_pnl"] * g["contracts"]).sum() / g["contracts"].sum(),
            }))
            .reset_index()
        )
        monthly["month"] = pd.to_datetime(monthly["month"])
        monthly = monthly.sort_values("month").reset_index(drop=True)
        monthly["tau"] = np.arange(len(monthly))
        monthly = monthly.dropna(subset=["maker_vwr"])

        # OLS: maker_vwr = γ_0 + γ_1 τ + ε
        ols = stats.linregress(monthly["tau"], monthly["maker_vwr"])
        gamma0, gamma1 = float(ols.intercept), float(ols.slope)
        gamma1_p  = float(ols.pvalue)
        gamma1_se = float(ols.stderr)
        r_sq      = float(ols.rvalue ** 2)
        monthly["fitted"] = gamma0 + gamma1 * monthly["tau"]

        # 95% CI band: SE_fit(τ) = s√(1/n + (τ−τ̄)²/Sxx), s²=SSR/(n−2)
        n = len(monthly)
        tau_arr  = monthly["tau"].values
        tau_mean = tau_arr.mean()
        s_xx = float(((tau_arr - tau_mean) ** 2).sum())
        ssr  = float(((monthly["maker_vwr"] - monthly["fitted"]) ** 2).sum())
        s    = np.sqrt(ssr / (n - 2))
        se_fit = s * np.sqrt(1 / n + (tau_arr - tau_mean) ** 2 / s_xx)
        t_crit = stats.t.ppf(0.975, df=n - 2)
        monthly["ci_lo"] = monthly["fitted"] - t_crit * se_fit
        monthly["ci_hi"] = monthly["fitted"] + t_crit * se_fit

        # ── 2. Quarterly maker + taker VWR ────────────────────────────────────
        with self.progress("Quarterly maker+taker VWR"):
            quarterly_dual = con.execute(f"""
                WITH resolved AS (
                    SELECT ticker, result
                    FROM '{self.markets_dir}/*.parquet'
                    WHERE status = 'finalized' AND result IN ('yes','no')
                )
                SELECT
                    DATE_TRUNC('quarter', t.created_time)            AS quarter,
                    SUM(({MAKER_PNL_SQL}) * t.count) / SUM(t.count) AS maker_vwr,
                    SUM(({TAKER_PNL_SQL}) * t.count) / SUM(t.count) AS taker_vwr
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN resolved m ON t.ticker = m.ticker
                GROUP BY quarter
                ORDER BY quarter
            """).df()
            quarterly_dual["quarter"] = pd.to_datetime(quarterly_dual["quarter"])

        # ── 3. Quarterly price-bucket stacked area ────────────────────────────
        with self.progress("Quarterly price-bucket stacked area"):
            bucket_raw = con.execute(f"""
                WITH resolved AS (
                    SELECT ticker, result
                    FROM '{self.markets_dir}/*.parquet'
                    WHERE status = 'finalized' AND result IN ('yes','no')
                )
                SELECT
                    DATE_TRUNC('quarter', t.created_time) AS quarter,
                    FLOOR(
                        CASE WHEN t.taker_side = 'yes'
                             THEN t.yes_price ELSE t.no_price END / 10
                    ) * 10 AS bucket_lo,
                    SUM(t.count) AS vol
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN resolved m ON t.ticker = m.ticker
                GROUP BY quarter, bucket_lo
                ORDER BY quarter, bucket_lo
            """).df()
            bucket_raw["quarter"] = pd.to_datetime(bucket_raw["quarter"])

        bucket_pivot = (
            bucket_raw.pivot_table(index="quarter", columns="bucket_lo", values="vol", aggfunc="sum")
            .fillna(0)
        )
        bucket_share = bucket_pivot.div(bucket_pivot.sum(axis=1), axis=0)

        summary = pd.DataFrame([
            {"metric": "gamma0_intercept",  "value": round(gamma0, 6)},
            {"metric": "gamma1_time_trend", "value": round(gamma1, 8)},
            {"metric": "gamma1_se",         "value": round(gamma1_se, 8)},
            {"metric": "gamma1_p_value",    "value": round(gamma1_p, 6)},
            {"metric": "r_squared",         "value": round(r_sq, 4)},
            {"metric": "n_months",          "value": n},
        ])

        fig = self._make_figure(
            monthly, quarterly_dual, bucket_share,
            gamma0, gamma1, gamma1_p, r_sq,
        )
        return AnalysisOutput(figure=fig, data=summary)

    def _make_figure(
        self,
        monthly: pd.DataFrame,
        quarterly_dual: pd.DataFrame,
        bucket_share: pd.DataFrame,
        g0: float,
        g1: float,
        p_val: float,
        r_sq: float,
    ) -> plt.Figure:
        from src.common.plot_style import (
            new_fig, clean_ax, stat_box, sig_stars,
            BLUE, GREEN, RED, GRAY,
        )

        fig = plt.figure(figsize=(16, 9))
        fig.suptitle("Kalshi — Temporal Dynamics of Market Efficiency",
                     fontsize=14, fontweight="semibold", y=0.98)
        gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.32,
                               left=0.07, right=0.97, top=0.91, bottom=0.10)
        ax_top   = fig.add_subplot(gs[0, :])
        ax_area  = fig.add_subplot(gs[1, 0])
        ax_trend = fig.add_subplot(gs[1, 1])

        # ── Top: Quarterly Maker vs Taker VWR ─────────────────────────────────
        q = quarterly_dual.sort_values("quarter").reset_index(drop=True)
        qlabels = q["quarter"].dt.to_period("Q").astype(str)
        x_idx = np.arange(len(q))

        ax_top.plot(x_idx, q["maker_vwr"] * 100, color=BLUE, lw=2.4, marker="o",
                    ms=5, label="Maker VWR")
        ax_top.plot(x_idx, q["taker_vwr"] * 100, color=GREEN, lw=2.4, marker="s",
                    ms=5, label="Taker VWR")
        ax_top.axhline(0, color=GRAY, lw=0.9, linestyle="--")
        ax_top.set_xticks(x_idx)
        ax_top.set_xticklabels(qlabels, rotation=35, ha="right", fontsize=8.5)
        clean_ax(ax_top, ylabel="Excess Return (%)",
                 title="Maker-Taker Returns Over Time", zero_h=False)
        ax_top.legend(fontsize=10, loc="upper right")

        # ── Bottom-left: Stacked area by price bucket ─────────────────────────
        quarters = bucket_share.index
        x_area = np.arange(len(quarters))
        qlabels_area = pd.to_datetime(quarters).to_period("Q").astype(str)
        bottom = np.zeros(len(quarters))
        cols_sorted = sorted(bucket_share.columns)

        for i, col in enumerate(cols_sorted):
            vals = bucket_share[col].values if col in bucket_share.columns else np.zeros(len(quarters))
            label = BUCKET_LABELS[i] if i < len(BUCKET_LABELS) else str(int(col))
            color = BUCKET_COLORS[i % len(BUCKET_COLORS)]
            ax_area.fill_between(x_area, bottom, bottom + vals * 100,
                                 alpha=0.88, color=color, linewidth=0, label=label)
            bottom = bottom + vals * 100

        ax_area.set_xlim(0, len(quarters) - 1)
        ax_area.set_ylim(0, 100)
        ax_area.set_xticks(x_area)
        ax_area.set_xticklabels(qlabels_area, rotation=35, ha="right", fontsize=7.5)
        clean_ax(ax_area, ylabel="Share of Taker Volume (%)",
                 title="Taker Volume Distribution by Price", zero_h=False)
        ax_area.legend(ncol=5, fontsize=7, loc="lower center",
                       bbox_to_anchor=(0.5, -0.42))

        # ── Bottom-right: Monthly OLS trend ───────────────────────────────────
        stars = sig_stars(p_val)
        sz = np.sqrt(monthly["n_contracts"].clip(1)) * 0.28 + 14
        ax_trend.scatter(monthly["month"], monthly["maker_vwr"] * 10_000,
                         s=sz, color=GREEN, edgecolors="#2a2a2a", linewidths=0.35,
                         alpha=0.78, zorder=3, label="Monthly VWR")
        ax_trend.plot(monthly["month"], monthly["fitted"] * 10_000,
                      color=RED, lw=2.2, zorder=4,
                      label=f"OLS  (γ₁={g1*1e4:.3f} bp/mo)")
        ax_trend.fill_between(monthly["month"],
                              monthly["ci_lo"] * 10_000,
                              monthly["ci_hi"] * 10_000,
                              alpha=0.13, color=RED, linewidth=0)
        clean_ax(ax_trend, xlabel="Month", ylabel="Maker VWR (basis points)",
                 title="Monthly OLS Trend", zero_h=False)
        ax_trend.legend(fontsize=8.5, loc="upper right")
        stat_box(ax_trend,
                 f"γ₁={g1*1e4:.3f} bp/mo {stars}\np={p_val:.4f}  R²={r_sq:.3f}",
                 loc="upper left")

        return fig
