"""CP05 — Cross-Sectional Heterogeneity: Kalshi.

OLS regression of trade-level Maker returns on category group dummies,
plus grouped bar chart showing maker vs taker VWR per category.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.kalshi.util.categories import category_sql, get_group
from src.common.analysis import Analysis, AnalysisOutput
from src.common.metrics import MAKER_PNL_SQL, TAKER_PNL_SQL, sharpe_ratio


class KalshiCategoryHeterogeneity(Analysis):
    """Category-level maker/taker return heterogeneity: OLS + grouped VWR breakdown."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="kalshi_cp05_category_heterogeneity",
            description="Cross-sectional maker vs taker return variation by category",
        )
        base = Path(__file__).parent.parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base / "data" / "kalshi" / "markets")

    def run(self) -> AnalysisOutput:
        con = duckdb.connect()

        with self.progress("Fetching maker+taker PnL with category"):
            df = con.execute(f"""
                WITH resolved AS (
                    SELECT ticker, event_ticker, result
                    FROM '{self.markets_dir}/*.parquet'
                    WHERE status = 'finalized' AND result IN ('yes','no')
                )
                SELECT
                    {category_sql("m")} AS category,
                    {MAKER_PNL_SQL}     AS maker_pnl,
                    {TAKER_PNL_SQL}     AS taker_pnl,
                    t.count             AS contracts
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN resolved m ON t.ticker = m.ticker
            """).df()

        df["group"] = df["category"].apply(get_group)

        # Per-group descriptive stats (maker + taker VWR)
        group_stats = []
        for grp, gdf in df.groupby("group"):
            w = gdf["contracts"]
            mr = gdf["maker_pnl"].dropna()
            tr = gdf["taker_pnl"].dropna()
            mw = w.loc[mr.index]
            tw = w.loc[tr.index]
            maker_vwr = float((mr * mw).sum() / mw.sum()) if mw.sum() > 0 else float("nan")
            taker_vwr = float((tr * tw).sum() / tw.sum()) if tw.sum() > 0 else float("nan")
            group_stats.append({
                "group":        grp,
                "n_contracts":  int(w.sum()),
                "maker_vwr":    maker_vwr,
                "taker_vwr":    taker_vwr,
                "sharpe":       sharpe_ratio(mr),
                "pct_positive": float((mr > 0).mean()) * 100,
            })
        cat_df = pd.DataFrame(group_stats).sort_values("n_contracts", ascending=False)

        # OLS with category dummies: R_j = β_0 + Σ β_g D_{j,g} + ε_j
        groups = cat_df["group"].tolist()
        reference = groups[0]
        others = groups[1:]

        sample = df[df["group"].isin(groups)].dropna(subset=["maker_pnl"])
        y = sample["maker_pnl"].values
        X = np.column_stack(
            [np.ones(len(sample))]
            + [(sample["group"] == g).astype(float).values for g in others]
        )
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        n_s, k = X.shape
        if n_s > k:
            sigma2 = float(np.sum((y - X @ coef) ** 2)) / (n_s - k)
            cov = sigma2 * np.linalg.pinv(X.T @ X)
            se = np.sqrt(np.diag(cov))
        else:
            se = np.full(len(coef), np.nan)
        t_vals = coef / np.clip(se, 1e-12, None)
        p_vals = 2 * stats.t.sf(np.abs(t_vals), df=max(n_s - k, 1))

        col_names = [f"β_{reference}(intercept)"] + [f"β_{g}" for g in others]
        ols_df = pd.DataFrame({
            "term":    col_names,
            "coef":    coef,
            "se":      se,
            "t_stat":  t_vals,
            "p_value": p_vals,
        })

        fig = self._make_figure(cat_df)
        return AnalysisOutput(figure=fig, data=cat_df, metadata={"ols": ols_df.to_dict("records")})

    def _make_figure(self, cat_df: pd.DataFrame) -> plt.Figure:
        from src.common.plot_style import (
            new_fig, clean_ax, stat_box, BLUE, GREEN, RED, GRAY,
        )

        top = cat_df.head(8).copy()
        grp_list = list(reversed(top["group"].tolist()))
        maker_vals = list(reversed((top["maker_vwr"] * 100).tolist()))
        taker_vals = list(reversed((top["taker_vwr"] * 100).tolist()))

        fig, ax = new_fig(1, 1, figsize=(12, 6),
                          suptitle="Kalshi — Maker vs Taker Returns by Category")

        y_idx = np.arange(len(grp_list))
        h = 0.36

        ax.barh(y_idx + h / 2, taker_vals, h,
                color=BLUE, alpha=0.88, linewidth=0, label="Taker VWR")
        ax.barh(y_idx - h / 2, maker_vals, h,
                color=GREEN, alpha=0.88, linewidth=0, label="Maker VWR")

        ax.set_yticks(y_idx)
        ax.set_yticklabels(grp_list, fontsize=10)
        ax.axvline(0, color=GRAY, lw=1.0, linestyle="--")
        clean_ax(ax, xlabel="Volume-Weighted Return (%)", zero_h=False, zero_v=False)
        ax.legend(fontsize=10, loc="lower right")

        fig.tight_layout()
        return fig
