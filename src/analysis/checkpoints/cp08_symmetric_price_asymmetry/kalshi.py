"""CP08 — Symmetric Price Asymmetry: Kalshi.

Three panels:
  1. Maker Excess Returns by Position Direction (1-cent granularity)
  2. Taker YES@p vs NO@p returns (grouped bars)
  3. YES − NO return gap scatter
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
from src.common.metrics import MAKER_PNL_SQL


class KalshiSymmetricPriceAsymmetry(Analysis):
    """Maker direction asymmetry + symmetric-price taker test."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
        bin_width: int = 5,
    ):
        super().__init__(
            name="kalshi_cp08_symmetric_price_asymmetry",
            description="Maker position-direction excess returns + taker YES/NO at equivalent implied prob",
        )
        base = Path(__file__).parent.parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base / "data" / "kalshi" / "markets")
        self.bin_width = bin_width

    def run(self) -> AnalysisOutput:
        con = duckdb.connect()

        # ── Maker Bought YES (taker_side='no') bucketed by yes_price at 1¢ ──
        with self.progress("Maker Bought YES returns"):
            maker_yes = con.execute(f"""
                WITH resolved AS (
                    SELECT ticker, result
                    FROM '{self.markets_dir}/*.parquet'
                    WHERE status = 'finalized' AND result IN ('yes','no')
                )
                SELECT
                    t.yes_price                                          AS price_cent,
                    SUM(({MAKER_PNL_SQL}) * t.count) / SUM(t.count)     AS maker_vwr,
                    SUM(t.count)                                         AS n_contracts
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN resolved m ON t.ticker = m.ticker
                WHERE t.taker_side = 'no'
                  AND t.yes_price BETWEEN 1 AND 99
                GROUP BY price_cent
                ORDER BY price_cent
            """).df()

        # ── Maker Bought NO (taker_side='yes') bucketed by no_price at 1¢ ───
        with self.progress("Maker Bought NO returns"):
            maker_no = con.execute(f"""
                WITH resolved AS (
                    SELECT ticker, result
                    FROM '{self.markets_dir}/*.parquet'
                    WHERE status = 'finalized' AND result IN ('yes','no')
                )
                SELECT
                    t.no_price                                           AS price_cent,
                    SUM(({MAKER_PNL_SQL}) * t.count) / SUM(t.count)     AS maker_vwr,
                    SUM(t.count)                                         AS n_contracts
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN resolved m ON t.ticker = m.ticker
                WHERE t.taker_side = 'yes'
                  AND t.no_price BETWEEN 1 AND 99
                GROUP BY price_cent
                ORDER BY price_cent
            """).df()

        # ── Taker: YES@p vs NO@p (binned) ─────────────────────────────────────
        with self.progress("Taker YES returns by bucket"):
            yes_df = con.execute(f"""
                WITH resolved AS (
                    SELECT ticker, result
                    FROM '{self.markets_dir}/*.parquet'
                    WHERE status = 'finalized' AND result IN ('yes','no')
                )
                SELECT
                    FLOOR(t.yes_price / {self.bin_width}) * {self.bin_width} AS price_bucket,
                    (CASE WHEN m.result = 'yes' THEN 1.0 ELSE 0.0 END) - t.yes_price / 100.0 AS taker_pnl,
                    t.count AS contracts
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN resolved m ON t.ticker = m.ticker
                WHERE t.taker_side = 'yes'
            """).df()

        with self.progress("Taker NO returns by bucket"):
            no_df = con.execute(f"""
                WITH resolved AS (
                    SELECT ticker, result
                    FROM '{self.markets_dir}/*.parquet'
                    WHERE status = 'finalized' AND result IN ('yes','no')
                )
                SELECT
                    FLOOR(t.no_price / {self.bin_width}) * {self.bin_width} AS price_bucket,
                    (CASE WHEN m.result = 'no' THEN 1.0 ELSE 0.0 END) - t.no_price / 100.0 AS taker_pnl,
                    t.count AS contracts
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN resolved m ON t.ticker = m.ticker
                WHERE t.taker_side = 'no'
            """).df()

        def bucket_vwr(df: pd.DataFrame) -> pd.Series:
            return (
                df.groupby("price_bucket")
                .apply(lambda g: (g["taker_pnl"] * g["contracts"]).sum() / g["contracts"].sum())
            )

        yes_vwr = bucket_vwr(yes_df).rename("yes_vwr")
        no_vwr  = bucket_vwr(no_df).rename("no_vwr")
        paired  = pd.concat([yes_vwr, no_vwr], axis=1).dropna().reset_index()
        paired["gap"] = paired["yes_vwr"] - paired["no_vwr"]

        if len(paired) >= 2:
            t_stat, p_val = stats.ttest_rel(paired["yes_vwr"], paired["no_vwr"])
        else:
            t_stat, p_val = float("nan"), float("nan")

        summary = paired.copy()
        summary["t_stat_gap"] = float(t_stat)
        summary["p_value_gap"] = float(p_val)

        fig = self._make_figure(maker_yes, maker_no, paired, float(t_stat), float(p_val))
        return AnalysisOutput(figure=fig, data=summary)

    def _make_figure(
        self,
        maker_yes: pd.DataFrame,
        maker_no: pd.DataFrame,
        paired: pd.DataFrame,
        t_stat: float,
        p_val: float,
    ) -> plt.Figure:
        from src.common.plot_style import (
            new_fig, clean_ax, stat_box, sig_stars,
            BLUE, GREEN, RED, GRAY,
        )

        fig = plt.figure(figsize=(16, 9))
        fig.suptitle("Kalshi — Symmetric Price Asymmetry",
                     fontsize=14, fontweight="semibold", y=0.98)
        gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.32,
                               left=0.07, right=0.97, top=0.91, bottom=0.09)
        ax_dir   = fig.add_subplot(gs[0, :])   # Maker direction — full width
        ax_bars  = fig.add_subplot(gs[1, 0])   # Taker YES vs NO grouped bars
        ax_gap   = fig.add_subplot(gs[1, 1])   # YES − NO gap scatter

        # ── Top: Maker Excess Returns by Position Direction ────────────────────
        ax_dir.plot(maker_yes["price_cent"], maker_yes["maker_vwr"] * 100,
                    color=BLUE, lw=1.5, alpha=0.88, label="Maker Bought YES")
        ax_dir.plot(maker_no["price_cent"], maker_no["maker_vwr"] * 100,
                    color=GREEN, lw=1.5, alpha=0.88, label="Maker Bought NO")
        ax_dir.axhline(0, color=GRAY, lw=0.9, linestyle="--")
        ax_dir.set_xlim(1, 99)
        ax_dir.set_xticks(range(1, 100, 2))
        ax_dir.set_xticklabels([str(x) for x in range(1, 100, 2)],
                               rotation=0, fontsize=7)
        clean_ax(ax_dir,
                 xlabel="Contract Price (cents)",
                 ylabel="Excess Return (%)",
                 title="Maker Excess Returns by Position Direction",
                 zero_h=False)
        ax_dir.legend(fontsize=10, loc="upper right")

        # ── Bottom-left: Taker YES@p vs NO@p grouped bars ─────────────────────
        x = np.arange(len(paired))
        w = 0.37
        ax_bars.bar(x - w / 2, paired["yes_vwr"] * 100, w,
                    label="YES @ p", color=GREEN, alpha=0.88, linewidth=0)
        ax_bars.bar(x + w / 2, paired["no_vwr"] * 100, w,
                    label="NO @ p",  color=RED,   alpha=0.88, linewidth=0)
        clean_ax(ax_bars,
                 xlabel="Implied Probability Bucket",
                 ylabel="Taker VW Return (pp)",
                 title="YES vs NO at Equivalent Implied Price")
        ax_bars.set_xticks(x)
        ax_bars.set_xticklabels([f"{int(b)}¢" for b in paired["price_bucket"]],
                                rotation=40, ha="right", fontsize=9)
        ax_bars.legend(fontsize=9)
        stars = sig_stars(p_val)
        stat_box(ax_bars, f"Paired t\nt = {t_stat:.2f},  p = {p_val:.4f} {stars}")

        # ── Bottom-right: YES − NO gap scatter ────────────────────────────────
        gap = paired["gap"] * 100
        pos = gap >= 0
        ax_gap.scatter(paired["price_bucket"][pos],  gap[pos],
                       color=GREEN, s=65, zorder=4, label="YES advantage")
        ax_gap.scatter(paired["price_bucket"][~pos], gap[~pos],
                       color=RED,   s=65, zorder=4, label="NO advantage")
        ax_gap.fill_between(paired["price_bucket"], gap, 0,
                            where=pos,  alpha=0.12, color=GREEN, linewidth=0)
        ax_gap.fill_between(paired["price_bucket"], gap, 0,
                            where=~pos, alpha=0.12, color=RED,   linewidth=0)
        ax_gap.axhline(0, color=GRAY, lw=1.0, linestyle="--")
        clean_ax(ax_gap,
                 xlabel="Implied Probability Bucket",
                 ylabel="YES − NO Return Gap (pp)",
                 title="Return Gap: YES vs NO at Same Implied Price",
                 zero_h=False)
        ax_gap.legend(fontsize=8.5)

        return fig
