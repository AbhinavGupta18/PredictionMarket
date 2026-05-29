# Prediction Market Microstructure

Empirical analysis of liquidity provision, behavioral biases, and market efficiency on binary prediction platforms (Kalshi and Polymarket).

Submitted to the **India Finance Conference (IFC)**.

## Structure

```
src/analysis/checkpoints/   # One folder per analysis (cp00a–cp13)
src/common/                 # Shared metrics, plotting, chart interfaces
output/                     # Generated figures and CSVs (gitignored)
data/                       # Raw parquet files — Kalshi & Polymarket (gitignored)
paper.tex                   # IEEE-format paper
```

## Checkpoints

| ID | Topic |
|----|-------|
| cp00a | Data validation |
| cp01 | Market overview |
| cp02 | Longshot bias |
| cp03 | Maker/taker PnL |
| cp04 | Spread analysis |
| cp05 | Category heterogeneity |
| cp06 | Temporal dynamics |
| cp07 | YES/NO asymmetry |
| cp08 | Symmetric price asymmetry |
| cp09 | Volume demand elasticity |
| cp10 | HHI concentration |
| cp11 | Category adverse selection |
| cp12 | Data limitations |
| cp13 | Sports sentiment arbitrage |

## Setup

```bash
uv sync
```

## Running

```bash
# Run all checkpoints
uv run run_checkpoints.py all

# Run a single checkpoint
uv run run_checkpoints.py cp03
```

Results land in `output/cp{NN}_{name}/kalshi/` and `/polymarket/`.
