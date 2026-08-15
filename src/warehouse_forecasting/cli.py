from __future__ import annotations

import argparse

from .data import DATASET_COLUMNS, demo_series, load_demand_series
from .experiment import run_experiment


ALL_MODELS = ["rnn", "lstm", "rnn-lstm", "gan", "mlp", "rbf", "arima", "arima-rbf"]


def parser():
    command = argparse.ArgumentParser(description="Warehouse demand forecasting — Mohd Ashraf Huzairie")
    source = command.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", help="CSV or Excel dataset path")
    source.add_argument("--demo", action="store_true", help="use a generated seasonal series")
    command.add_argument("--dataset", choices=DATASET_COLUMNS, default="store_item")
    command.add_argument("--models", nargs="+", choices=ALL_MODELS, default=["mlp", "rbf"])
    command.add_argument("--output", default="artifacts")
    command.add_argument("--lookback", type=int, default=30)
    command.add_argument("--folds", type=int, default=5)
    command.add_argument("--epochs", type=int, default=40)
    command.add_argument("--seed", type=int, default=42)
    return command


def main():
    args = parser().parse_args()
    series = demo_series(seed=args.seed) if args.demo else load_demand_series(args.data, args.dataset)
    summary = run_experiment(series, args.models, args.output, lookback=args.lookback, n_splits=args.folds, seed=args.seed, epochs=args.epochs)
    print(summary.to_string(index=False))
