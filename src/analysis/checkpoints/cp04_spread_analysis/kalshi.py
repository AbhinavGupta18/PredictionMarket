"""CP04 — Spread Compensation Analysis: Kalshi.

Tests whether Maker profitability exceeds the bid-ask spread, decomposing
α_Maker = S_t - ΔP_{t+Δt} and running a one-sided t-test H0: Maker return ≤ half-spread.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.common.analysis import Analysis, AnalysisOutput
from src.common.metrics import MAKER_PNL_SQL


class KalshiSpreadAnalysis(Analysis):
    """Spread decomposition: is maker alpha more than just spread capture?"""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="kalshi_cp04_spread_analysis",
            description="Disentangle maker alpha from bid-ask spread compensation",
        )
        base = Path(__file__).parent.parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base / "data" / "kalshi" / "markets")

    def run(self) -> AnalysisOutput:
        con = duckdb.connect()

        with self.progress("Loading maker PnL per trade"):
            pnl_df = con.execute(f"""
                WITH resolved AS (
                    SELECT ticker, result
                    FROM '{self.markets_dir}/*.parquet'
                    WHERE status = 'finalized' AND result IN ('yes','no')
                )
                SELECT
                    t.ticker,
                    t.yes_price,
                    t.created_time,
                    {MAKER_PNL_SQL}    AS maker_pnl,
                    t.count            AS contracts
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN resolved m ON t.ticker = m.ticker
                ORDER BY t.ticker, t.created_time
            """).df()

        with self.progress("Estimating bid-ask spread via consecutive-trade price changes"):
            # Within each ticker, compute |price change| between successive trades
            pnl_df["created_time"] = pd.to_datetime(pnl_df["created_time"])
            pnl_df = pnl_df.sort_values(["ticker", "created_time"])
            pnl_df["prev_price"] = pnl_df.groupby("ticker")["yes_price"].shift(1)
            pnl_df["price_change"] = (pnl_df["yes_price"] - pnl_df["prev_price"]).abs()

        spread_df = pnl_df.dropna(subset=["price_change"])

        # Estimated effective half-spread = avg |price change| / 2 (in cents → fraction)
        avg_abs_change = float(spread_df["price_change"].mean())
        estimated_half_spread = avg_abs_change / 2.0 / 100.0

        # Volume-weighted maker return
        maker_vwr = float(
            (pnl_df["maker_pnl"] * pnl_df["contracts"]).sum()
            / pnl_df["contracts"].sum()
        )

        # One-sided t-test: H0 maker_pnl <= half_spread; H1 maker_pnl > half_spread
        maker_pnl_vals = pnl_df["maker_pnl"].dropna().values
        t_stat, p_two = stats.ttest_1samp(maker_pnl_vals, estimated_half_spread)
        # One-sided p (right tail)
        p_one_sided = float(p_two) / 2 if float(t_stat) > 0 else 1.0 - float(p_two) / 2

        # Rolling 30-day maker VWR
        pnl_df["date"] = pnl_df["created_time"].dt.date
        daily = (
            pnl_df.groupby("date")
            .apply(lambda g: (g["maker_pnl"] * g["contracts"]).sum() / g["contracts"].sum())
            .reset_index(name="maker_vwr")
        )
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.sort_values("date")
        daily["rolling30"] = daily["maker_vwr"].rolling(30, min_periods=5).mean()

        summary = pd.DataFrame([
            {"metric": "maker_vw_return",      "value": round(maker_vwr, 6)},
            {"metric": "est_half_spread",      "value": round(estimated_half_spread, 6)},
            {"metric": "alpha_net_of_spread",  "value": round(maker_vwr - estimated_half_spread, 6)},
            {"metric": "t_stat_vs_spread",     "value": round(float(t_stat), 4)},
            {"metric": "p_one_sided",          "value": round(p_one_sided, 6)},
            {"metric": "avg_abs_price_change_cents", "value": round(avg_abs_change, 4)},
        ])

        fig = self._make_figure(daily, maker_vwr, estimated_half_spread, float(t_stat), p_one_sided)
        return AnalysisOutput(figure=fig, data=summary)

    def _make_figure(
        self,
        daily: pd.DataFrame,
        maker_vwr: float,
        half_spread: float,
        t_stat: float,
        p_one: float,
    ) -> plt.Figure:
        from src.common.plot_style import (
            new_fig, clean_ax, stat_box, sig_stars, shade_v,
            GREEN, RED, BLUE, AMBER, GRAY,
        )

        fig, axes = new_fig(1, 2, suptitle="Kalshi — Spread Compensation Analysis")
        stars = sig_stars(p_one)
        net_alpha = maker_vwr - half_spread

        # ── Left: rolling maker return vs half-spread ─────────────────────────
        ax = axes[0]
        # Shade region where maker earns above spread
        shade_v(ax, half_spread * 1e4, max(daily["rolling30"].dropna()) * 1e4 * 1.3,
                color=GREEN, alpha=0.08)
        ax.plot(daily["date"], daily["rolling30"] * 10_000,
                color=GREEN, lw=1.6, label="30-day rolling maker return")
        ax.axhline(half_spread * 10_000, color=RED, lw=1.6, linestyle="--",
                   label=f"Est. half-spread  ({half_spread*100:.2f}¢)")
        clean_ax(ax,
                 xlabel="Date",
                 ylabel="Return (basis points)",
                 title="Rolling Maker Return vs Estimated Half-Spread")
        ax.legend()
        stat_box(ax, f"Net α = {net_alpha*1e4:.1f} bp  {stars}", loc="upper left")

        # ── Right: decomposition bar chart ────────────────────────────────────
        ax2 = axes[1]
        labels = ["Maker VWR", "Half-Spread", "Net Alpha"]
        vals   = [maker_vwr * 1e4, half_spread * 1e4, net_alpha * 1e4]
        clrs   = [GREEN, RED, BLUE if net_alpha >= 0 else RED]
        bars   = ax2.bar(labels, vals, color=clrs, width=0.48, alpha=0.9, linewidth=0)
        ax2.bar_label(bars, fmt="%.1f bp", padding=4, fontsize=9.5)
        clean_ax(ax2,
                 ylabel="Return (basis points)",
                 title="Maker Return Decomposition")
        stat_box(ax2,
                 f"H₀: Maker ≤ half-spread\nt = {t_stat:.2f},  p = {p_one:.4f} {stars}",
                 loc="lower right")

        fig.tight_layout()
        return fig
