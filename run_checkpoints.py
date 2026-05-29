"""Run checkpoint analyses for the prediction market microstructure paper.

Usage:
    uv run run_checkpoints.py all              # Run all checkpoints
    uv run run_checkpoints.py cp01             # Run a single checkpoint
    uv run run_checkpoints.py cp02 cp05 cp07   # Run multiple checkpoints
    uv run run_checkpoints.py list             # List available checkpoints
"""

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path

from src.common.analysis import Analysis

BASE = "src.analysis.checkpoints"

CHECKPOINTS: dict[str, list[tuple[str, str]]] = {
    "cp00a": [
        ("cp00a_data_validation", f"{BASE}.cp00a_data_validation.kalshi"),
        ("cp00a_data_validation", f"{BASE}.cp00a_data_validation.polymarket"),
    ],
    "cp01": [
        ("cp01_market_overview", f"{BASE}.cp01_market_overview.kalshi"),
        ("cp01_market_overview", f"{BASE}.cp01_market_overview.polymarket"),
    ],
    "cp02": [
        ("cp02_longshot_bias", f"{BASE}.cp02_longshot_bias.kalshi"),
        ("cp02_longshot_bias", f"{BASE}.cp02_longshot_bias.polymarket"),
    ],
    "cp03": [
        ("cp03_maker_taker_pnl", f"{BASE}.cp03_maker_taker_pnl.kalshi"),
    ],
    "cp04": [
        ("cp04_spread_analysis", f"{BASE}.cp04_spread_analysis.kalshi"),
    ],
    "cp05": [
        ("cp05_category_heterogeneity", f"{BASE}.cp05_category_heterogeneity.kalshi"),
    ],
    "cp06": [
        ("cp06_temporal_dynamics", f"{BASE}.cp06_temporal_dynamics.kalshi"),
        ("cp06_temporal_dynamics", f"{BASE}.cp06_temporal_dynamics.polymarket"),
    ],
    "cp07": [
        ("cp07_yes_no_asymmetry", f"{BASE}.cp07_yes_no_asymmetry.kalshi"),
    ],
    "cp08": [
        ("cp08_symmetric_price_asymmetry", f"{BASE}.cp08_symmetric_price_asymmetry.kalshi"),
    ],
    "cp09": [
        ("cp09_volume_demand_elasticity", f"{BASE}.cp09_volume_demand_elasticity.kalshi"),
    ],
    "cp10": [
        ("cp10_hhi_concentration", f"{BASE}.cp10_hhi_concentration.kalshi"),
        ("cp10_hhi_concentration", f"{BASE}.cp10_hhi_concentration.polymarket"),
    ],
    "cp11": [
        ("cp11_category_adverse_selection", f"{BASE}.cp11_category_adverse_selection.kalshi"),
    ],
    "cp12": [
        ("cp12_data_limitations", f"{BASE}.cp12_data_limitations.kalshi"),
        ("cp12_data_limitations", f"{BASE}.cp12_data_limitations.polymarket"),
    ],
    "cp13": [
        ("cp13_sports_sentiment_arbitrage", f"{BASE}.cp13_sports_sentiment_arbitrage.kalshi"),
    ],
}


def _load_analysis_classes(module_path: str) -> list[type[Analysis]]:
    module = importlib.import_module(module_path)
    classes = []
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, Analysis) and obj is not Analysis and not inspect.isabstract(obj):
            classes.append(obj)
    return classes


def run_checkpoint(key: str) -> None:
    """Run all analyses registered under a checkpoint key."""
    key = key.lower()
    if key not in CHECKPOINTS:
        print(f"Unknown checkpoint: {key}. Run 'list' to see options.")
        sys.exit(1)

    entries = CHECKPOINTS[key]
    sep = "=" * 60
    print(f"\n{sep}\n  {key.upper()}\n{sep}")

    for folder_name, module_path in entries:
        platform = module_path.rsplit(".", 1)[-1]
        output_dir = Path("output") / folder_name / platform
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            classes = _load_analysis_classes(module_path)
        except (ImportError, ModuleNotFoundError) as exc:
            print(f"  [ERROR] Failed to load {module_path}: {exc}")
            continue

        for cls in classes:
            instance = cls()
            print(f"\n  [{platform.upper()}] {instance.name}")
            saved = instance.save(output_dir, formats=["png", "pdf", "csv", "json"])
            for fmt, path in saved.items():
                print(f"    [{fmt.upper():4s}] {path}")


def main() -> None:
    """Entry point: parse CLI args and dispatch to run_checkpoint."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == "list":
        print("\nAvailable checkpoints:\n")
        for key, entries in CHECKPOINTS.items():
            platforms = ", ".join(e[1].rsplit(".", 1)[-1] for e in entries)
            print(f"  {key:6s}  [{platforms}]  ->  output/{entries[0][0]}/")
        print()
        return

    keys = list(CHECKPOINTS.keys()) if command == "all" else sys.argv[1:]

    for key in keys:
        run_checkpoint(key.lower())

    print("\nDone. Results in output/\n")


if __name__ == "__main__":
    main()
