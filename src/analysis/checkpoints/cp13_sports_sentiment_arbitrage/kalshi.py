"""CP13 — Sports Sentiment Arbitrage: Kalshi.

Decomposes sports maker-alpha into three complementary lenses:
  1. By sport (NFL, NBA, MLB, Soccer, Tennis, Golf, UFC, Racing, NCAA, NHL)
  2. Maker VWR vs time-to-resolution (sentiment decay)
  3. YES-side fan bias per sport

Uses the same aggregation pattern as CP05 (no SQL-level sports filter) to
guarantee consistent results with the known 12.8B sports contract baseline.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.kalshi.util.categories import SUBCATEGORY_PATTERNS, get_group
from src.common.analysis import Analysis, AnalysisOutput
from src.common.metrics import MAKER_PNL_SQL, TAKER_PNL_SQL

# ── Sport-subcategory lookup ───────────────────────────────────────────────────
# Maps extracted event_ticker prefix → sport label using SUBCATEGORY_PATTERNS.
_SPORT_PATTERNS: list[tuple[str, str]] = [
    (p, cat)
    for p, grp, cat, _ in SUBCATEGORY_PATTERNS
    if grp == "Sports"
]


def _get_sport(prefix: str) -> str:
    pu = prefix.upper()
    for pattern, sport in _SPORT_PATTERNS:
        if pattern in pu:
            return sport
    return "Other Sports"


# Time-to-resolution bucket labels (ascending proximity to event)
_TTR_LABELS = ["30 d+", "7–30 d", "3–7 d", "1–3 d", "6 h–1 d", "0–6 h"]


class KalshiSportsSentimentArbitrage(Analysis):
    """Sports sentiment arbitrage: maker alpha by sport, TTR decay, YES-bias."""

    def __init__(
        self,
        trades_dir:  Path | str | None = None,
        markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="kalshi_cp13_sports_sentiment_arbitrage",
            description=(
                "Sports sentiment arbitrage: maker alpha by sport, "
                "time-to-resolution decay, YES-side fan bias"
            ),
        )
        base = Path(__file__).parent.parent.parent.parent.parent
        self.trades_dir  = Path(trades_dir  or base / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base / "data" / "kalshi" / "markets")

    def run(self) -> AnalysisOutput:
        con = duckdb.connect()

        # ── Query A: aggregate ALL categories by prefix + side ────────────────
        # Same join pattern as CP05 — no sports filter in SQL.
        # Group by (prefix, side) → tiny result; filter to Sports in Python.
        with self.progress("All-category PnL aggregation by prefix + side"):
            raw_a = con.execute(f"""
                WITH resolved AS (
                    SELECT ticker, event_ticker, result
                    FROM '{self.markets_dir}/*.parquet'
                    WHERE status = 'finalized'
                      AND result IN ('yes', 'no')
                )
                SELECT
                    regexp_extract(m.event_ticker, '^([A-Z0-9]+)', 1) AS cat_prefix,
                    t.taker_side,
                    SUM(({MAKER_PNL_SQL}) * t.count)                  AS maker_pnl_w,
                    SUM(({TAKER_PNL_SQL}) * t.count)                  AS taker_pnl_w,
                    SUM(t.count)                                       AS n_contracts
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN resolved m ON t.ticker = m.ticker
                GROUP BY cat_prefix, t.taker_side
            """).df()

        # Keep only Sports-group prefixes
        raw_a["group"] = raw_a["cat_prefix"].apply(get_group)
        raw_a = raw_a[raw_a["group"] == "Sports"].copy()
        raw_a["sport"] = raw_a["cat_prefix"].apply(_get_sport)

        # Aggregate per sport (sum pre-weighted PnL over prefixes and sides)
        sport_stats: list[dict] = []
        for sport, gdf in raw_a.groupby("sport"):
            n_total = float(gdf["n_contracts"].sum())
            if n_total == 0:
                continue
            maker_vwr = float(gdf["maker_pnl_w"].sum() / n_total)
            taker_vwr = float(gdf["taker_pnl_w"].sum() / n_total)
            yes_vol   = float(
                gdf.loc[gdf["taker_side"] == "yes", "n_contracts"].sum()
            )
            sport_stats.append({
                "sport":       sport,
                "n_contracts": int(n_total),
                "maker_vwr":   maker_vwr,
                "taker_vwr":   taker_vwr,
                "yes_share":   yes_vol / n_total,
            })

        sport_df = (
            pd.DataFrame(sport_stats)
            .sort_values("n_contracts", ascending=False)
            .head(10)
            .reset_index(drop=True)
        )

        # ── Query B: ALL categories, TTR-bucketed PnL (filter in Python) ──────
        # Computing TTR for all trades then filtering avoids the unreliable
        # event_ticker LIKE filter; group by (prefix, ttr_bucket) in SQL.
        with self.progress("Time-to-resolution maker VWR (all cats → filter Sports)"):
            raw_b = con.execute(f"""
                WITH resolved AS (
                    SELECT ticker, event_ticker, result,
                           CAST(close_time AS TIMESTAMPTZ) AS close_ts
                    FROM '{self.markets_dir}/*.parquet'
                    WHERE status = 'finalized'
                      AND result IN ('yes', 'no')
                      AND close_time < '2099-01-01'
                )
                SELECT
                    regexp_extract(m.event_ticker, '^([A-Z0-9]+)', 1) AS cat_prefix,
                    CASE
                        WHEN GREATEST(
                                (epoch(m.close_ts)
                                 - epoch(CAST(t.created_time AS TIMESTAMPTZ)))
                                / 3600.0, 0.0) <    6 THEN '0–6 h'
                        WHEN GREATEST(
                                (epoch(m.close_ts)
                                 - epoch(CAST(t.created_time AS TIMESTAMPTZ)))
                                / 3600.0, 0.0) <   24 THEN '6 h–1 d'
                        WHEN GREATEST(
                                (epoch(m.close_ts)
                                 - epoch(CAST(t.created_time AS TIMESTAMPTZ)))
                                / 3600.0, 0.0) <   72 THEN '1–3 d'
                        WHEN GREATEST(
                                (epoch(m.close_ts)
                                 - epoch(CAST(t.created_time AS TIMESTAMPTZ)))
                                / 3600.0, 0.0) <  168 THEN '3–7 d'
                        WHEN GREATEST(
                                (epoch(m.close_ts)
                                 - epoch(CAST(t.created_time AS TIMESTAMPTZ)))
                                / 3600.0, 0.0) <  720 THEN '7–30 d'
                        ELSE                               '30 d+'
                    END                                                AS ttr_bucket,
                    SUM(({MAKER_PNL_SQL}) * t.count)                   AS maker_pnl_w,
                    SUM(t.count)                                       AS n_contracts
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN resolved m ON t.ticker = m.ticker
                GROUP BY cat_prefix, ttr_bucket
            """).df()

        # Filter to Sports and aggregate across prefixes per TTR bucket
        raw_b["group"] = raw_b["cat_prefix"].apply(get_group)
        sports_b = raw_b[raw_b["group"] == "Sports"].copy()

        ttr_agg = (
            sports_b.groupby("ttr_bucket", as_index=False)
            .agg(maker_pnl_w=("maker_pnl_w", "sum"),
                 n_contracts=("n_contracts", "sum"))
        )
        ttr_agg["maker_vwr"] = ttr_agg["maker_pnl_w"] / ttr_agg["n_contracts"]
        ttr_agg["ttr_bucket"] = pd.Categorical(
            ttr_agg["ttr_bucket"], categories=list(reversed(_TTR_LABELS)), ordered=True
        )
        ttr_df = ttr_agg.sort_values("ttr_bucket").reset_index(drop=True)

        # OLS: does maker_vwr trend from far-out to close-in?
        ttr_clean = ttr_df.dropna(subset=["maker_vwr"])
        if len(ttr_clean) >= 3:
            ols       = stats.linregress(
                np.arange(len(ttr_clean)), ttr_clean["maker_vwr"].values
            )
            ttr_slope = float(ols.slope)
            ttr_p     = float(ols.pvalue)
            ttr_r2    = float(ols.rvalue ** 2)
        else:
            ttr_slope = ttr_p = ttr_r2 = float("nan")

        summary = sport_df.copy()
        summary["ttr_slope_per_bin"] = ttr_slope
        summary["ttr_p_value"]       = ttr_p
        summary["ttr_r2"]            = ttr_r2

        fig = self._make_figure(sport_df, ttr_df, ttr_slope, ttr_p, ttr_r2)
        return AnalysisOutput(figure=fig, data=summary)

    def _make_figure(
        self,
        sport_df:  pd.DataFrame,
        ttr_df:    pd.DataFrame,
        ttr_slope: float,
        ttr_p:     float,
        ttr_r2:    float,
    ) -> plt.Figure:
        from src.common.plot_style import (
            clean_ax, stat_box, sig_stars,
            BLUE, GREEN, RED, GRAY, AMBER,
        )

        fig = plt.figure(figsize=(16, 9))
        fig.suptitle(
            "Kalshi Sports Markets — Sentiment Arbitrage Decomposition",
            fontsize=14, fontweight="semibold", y=0.98,
        )
        gs = gridspec.GridSpec(
            2, 2, hspace=0.46, wspace=0.33,
            left=0.07, right=0.97, top=0.91, bottom=0.10,
        )
        ax_sport = fig.add_subplot(gs[0, :])
        ax_ttr   = fig.add_subplot(gs[1, 0])
        ax_yes   = fig.add_subplot(gs[1, 1])

        # ── Top: grouped horizontal bars — maker & taker VWR by sport ─────────
        sdf    = sport_df.sort_values("n_contracts", ascending=True)
        labels = sdf["sport"].tolist()
        mkv    = (sdf["maker_vwr"] * 100).tolist()
        tkv    = (sdf["taker_vwr"] * 100).tolist()
        y      = np.arange(len(labels))
        h      = 0.36

        ax_sport.barh(y + h / 2, tkv, h,
                      color=BLUE,  alpha=0.88, linewidth=0, label="Taker VWR")
        ax_sport.barh(y - h / 2, mkv, h,
                      color=GREEN, alpha=0.88, linewidth=0, label="Maker VWR")
        ax_sport.set_yticks(y)
        ax_sport.set_yticklabels(labels, fontsize=9)
        ax_sport.axvline(0, color=GRAY, lw=1.0, linestyle="--")
        clean_ax(ax_sport,
                 xlabel="Volume-Weighted Return (%)",
                 title="Maker vs Taker VWR by Sport",
                 zero_h=False, zero_v=False)
        ax_sport.legend(fontsize=9, loc="lower right")

        # Volume annotation
        xlim = ax_sport.get_xlim()
        for i, row in enumerate(sdf.itertuples()):
            vol_b = row.n_contracts / 1e9
            ax_sport.text(
                xlim[1] * 0.99, y[i], f"{vol_b:.1f}B",
                va="center", ha="right", fontsize=7.5, color=GRAY,
            )

        # ── Bottom-left: TTR bar chart + OLS trend ────────────────────────────
        ttr_plot = ttr_df.dropna(subset=["maker_vwr"])
        x_ttr    = np.arange(len(ttr_plot))
        bar_c    = [GREEN if v >= 0 else RED for v in ttr_plot["maker_vwr"]]

        ax_ttr.bar(x_ttr, ttr_plot["maker_vwr"] * 100,
                   color=bar_c, alpha=0.88, linewidth=0)

        if not np.isnan(ttr_slope) and len(ttr_plot) >= 3:
            fit = np.poly1d(np.polyfit(x_ttr, ttr_plot["maker_vwr"].values, 1))
            ax_ttr.plot(x_ttr, fit(x_ttr) * 100,
                        color=AMBER, lw=2.0, linestyle="--",
                        zorder=5, label="OLS trend")
            ax_ttr.legend(fontsize=8, loc="upper right")

        ax_ttr.axhline(0, color=GRAY, lw=1.0, linestyle="--")
        ax_ttr.set_xticks(x_ttr)
        ax_ttr.set_xticklabels(ttr_plot["ttr_bucket"].tolist(),
                               rotation=28, ha="right", fontsize=8.5)
        clean_ax(ax_ttr,
                 xlabel="Time Before Resolution",
                 ylabel="Maker VWR (%)",
                 title="Sentiment Decay: Maker Alpha vs Time-to-Resolution",
                 zero_h=False)
        if not np.isnan(ttr_slope):
            stars = sig_stars(ttr_p)
            stat_box(ax_ttr,
                     f"OLS slope: {ttr_slope*100:.4f}%/bin {stars}\n"
                     f"p = {ttr_p:.4f}   R² = {ttr_r2:.3f}",
                     loc="upper right")

        # ── Bottom-right: YES capital share by sport ───────────────────────────
        ydf     = sport_df.sort_values("n_contracts", ascending=True)
        y_yes   = np.arange(len(ydf))
        yes_val = ydf["yes_share"].values * 100
        bar_cy  = [GREEN if s >= 50 else RED for s in yes_val]

        ax_yes.barh(y_yes, yes_val, color=bar_cy, alpha=0.88, linewidth=0)
        ax_yes.axvline(50, color=GRAY, lw=1.3, linestyle="--", label="50% neutral")
        ax_yes.set_yticks(y_yes)
        ax_yes.set_yticklabels(ydf["sport"].tolist(), fontsize=9)
        ax_yes.set_xlim(30, 80)
        clean_ax(ax_yes,
                 xlabel="YES Capital Share (%)",
                 title="Fan Enthusiasm: YES-Side Bias by Sport",
                 zero_h=False, zero_v=False)
        ax_yes.legend(fontsize=8.5, loc="upper left")

        return fig
