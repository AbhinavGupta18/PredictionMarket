"""CP07 — YES/NO Behavioral Asymmetry: Kalshi.

Tests whether Takers show a systematic preference for affirmative (YES) bets
over negation (NO) bets, comparing E[Return | Side=YES] vs E[Return | Side=NO].
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.common.analysis import Analysis, AnalysisOutput
from src.common.metrics import TAKER_PNL_SQL


class KalshiYesNoAsymmetry(Analysis):
    """YES/NO directional bias: returns and volume by taker side."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="kalshi_cp07_yes_no_asymmetry",
            description="Affirmative bias: compare taker returns for YES vs NO side",
        )
        base = Path(__file__).parent.parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base / "data" / "kalshi" / "markets")

    def run(self) -> AnalysisOutput:
        con = duckdb.connect()

        with self.progress("Computing taker returns by side"):
            df = con.execute(f"""
                WITH resolved AS (
                    SELECT ticker, result
                    FROM '{self.markets_dir}/*.parquet'
                    WHERE status = 'finalized' AND result IN ('yes','no')
                )
                SELECT
                    DATE_TRUNC('month', t.created_time)                         AS month,
                    t.taker_side,
                    {TAKER_PNL_SQL}                                             AS taker_pnl,
                    CASE WHEN t.taker_side = 'yes' THEN t.yes_price
                         ELSE t.no_price END                                    AS price_cents,
                    t.count                                                     AS contracts
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN resolved m ON t.ticker = m.ticker
            """).df()

        yes_df = df[df["taker_side"] == "yes"]
        no_df  = df[df["taker_side"] == "no"]

        def vwr(g: pd.DataFrame) -> float:
            w = g["contracts"]
            return float((g["taker_pnl"] * w).sum() / w.sum())

        yes_vwr = vwr(yes_df)
        no_vwr  = vwr(no_df)

        # Paired t-test on monthly VWR differences: YES_VWR_t − NO_VWR_t
        # Tests H0: E[monthly YES return] = E[monthly NO return], controls for
        # market-wide variation within each month
        monthly_side = (
            df.groupby(["month", "taker_side"])
            .apply(lambda g: (g["taker_pnl"] * g["contracts"]).sum() / g["contracts"].sum())
            .unstack("taker_side")
        ).dropna()
        if "yes" in monthly_side.columns and "no" in monthly_side.columns and len(monthly_side) >= 2:
            t_stat, p_val = stats.ttest_rel(
                monthly_side["yes"].values,
                monthly_side["no"].values,
            )
        else:
            t_stat, p_val = float("nan"), float("nan")

        # Volume split
        total_contracts = df["contracts"].sum()
        yes_vol_share = float(yes_df["contracts"].sum() / total_contracts) * 100
        no_vol_share  = float(no_df["contracts"].sum()  / total_contracts) * 100

        # By price bucket
        df["bucket"] = (df["price_cents"] // 10) * 10
        bucket = (
            df.groupby(["bucket", "taker_side"])
            .apply(lambda g: pd.Series({
                "vwr": (g["taker_pnl"] * g["contracts"]).sum() / g["contracts"].sum(),
                "n_contracts": g["contracts"].sum(),
            }))
            .reset_index()
        )

        summary = pd.DataFrame([
            {"metric": "yes_vw_return",    "value": round(yes_vwr, 6)},
            {"metric": "no_vw_return",     "value": round(no_vwr, 6)},
            {"metric": "yes_no_gap",       "value": round(yes_vwr - no_vwr, 6)},
            {"metric": "t_stat",           "value": round(float(t_stat), 4)},
            {"metric": "p_value",          "value": round(float(p_val), 6)},
            {"metric": "yes_volume_pct",   "value": round(yes_vol_share, 2)},
            {"metric": "no_volume_pct",    "value": round(no_vol_share, 2)},
            {"metric": "yes_n_trades",     "value": len(yes_df)},
            {"metric": "no_n_trades",      "value": len(no_df)},
        ])

        fig = self._make_figure(bucket, yes_vwr, no_vwr, float(t_stat), float(p_val),
                                yes_vol_share, no_vol_share)
        return AnalysisOutput(figure=fig, data=summary)

    def _make_figure(
        self,
        bucket: pd.DataFrame,
        yes_vwr: float,
        no_vwr: float,
        t_stat: float,
        p_val: float,
        yes_vol: float,
        no_vol: float,
    ) -> plt.Figure:
        from src.common.plot_style import (
            new_fig, clean_ax, stat_box, sig_stars, GREEN, RED, GRAY, BLUE,
        )

        fig, axes = new_fig(1, 2, suptitle="Kalshi — YES/NO Behavioral Asymmetry")
        stars = sig_stars(p_val)

        # ── Left: VWR by bucket and side ─────────────────────────────────────
        yes_b = bucket[bucket["taker_side"] == "yes"].set_index("bucket")["vwr"]
        no_b  = bucket[bucket["taker_side"] == "no"].set_index("bucket")["vwr"]
        common_idx = sorted(set(yes_b.index) & set(no_b.index))
        x = np.arange(len(common_idx))
        w = 0.37
        axes[0].bar(x - w / 2, yes_b.loc[common_idx] * 100, w,
                    label="YES Taker", color=GREEN, alpha=0.88, linewidth=0)
        axes[0].bar(x + w / 2, no_b.loc[common_idx] * 100, w,
                    label="NO Taker",  color=RED,   alpha=0.88, linewidth=0)
        clean_ax(axes[0],
                 xlabel="Price Bucket",
                 ylabel="Volume-Weighted Return (pp)",
                 title="Taker Return by Side & Price Bucket")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([f"{b}¢" for b in common_idx], fontsize=9)
        axes[0].legend()
        stat_box(axes[0], f"Paired monthly t-test\nt = {t_stat:.2f},  p = {p_val:.4f} {stars}")

        # ── Right: diverging bar (volume split + VWR) ─────────────────────────
        ax2 = axes[1]
        categories = ["YES Taker", "NO Taker"]
        vwrs  = [yes_vwr * 100, no_vwr * 100]
        vols  = [yes_vol, no_vol]
        clrs  = [GREEN, RED]
        bars = ax2.barh(categories, vwrs, color=clrs, alpha=0.88, linewidth=0, height=0.42)
        ax2.bar_label(bars, fmt="%.3f pp", padding=5, fontsize=9.5)
        clean_ax(ax2,
                 xlabel="Volume-Weighted Return (pp)",
                 title="Aggregate VWR: YES vs NO",
                 zero_h=False, zero_v=True)
        # Annotate volume share
        for i, (cat, vol) in enumerate(zip(categories, vols)):
            ax2.text(0, i, f"  {vol:.1f}% of volume",
                     va="center", fontsize=8.5, color=GRAY)

        fig.tight_layout()
        return fig
