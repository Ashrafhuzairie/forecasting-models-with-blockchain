# Comparative Forecasting Models for Automated Warehouse Replenishment

Author: **Mohd Ashraf Huzairie**

This repository implements the experimental workflow described in the IEEE paper *Comparative Performance Analysis of Forecasting Models for Automated Warehouse Replenishment*. It provides a consistent pipeline for comparing RNN, LSTM, hybrid RNN-LSTM, GAN, MLP, RBF, ARIMA, and hybrid ARIMA-RBF forecasts on warehouse-demand time series.

## What is included

- adapters for the three paper datasets (retail supply-chain sales, historical product demand, and the store-item demand forecasting dataset);
- cleaning, daily aggregation, chronological splitting, lag-window creation, and Min-Max scaling;
- time-aware K-fold validation with MSE, RMSE, MAE, R2, and MAPE;
- paper-result tables for reproducible comparison with the published results;
- an optional hash-chained JSON ledger recording experiment configurations and results.

## Quick start

Requires Python 3.10 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
python main.py --demo --models mlp rbf arima --output artifacts
```

Deep-learning models require the optional TensorFlow dependency:

```bash
pip install -e ".[deep-learning]"
python main.py --demo --models rnn lstm rnn-lstm gan
```

For the optional web view, install `pip install -e ".[web]"` and run `python app.py`.

To run against a downloaded Kaggle file:

```bash
python main.py --data path/to/file.csv --dataset store_item --models lstm rnn-lstm
```

Supported dataset identifiers are `retail_supply_chain`, `historical_product_demand`, and `store_item`. Dataset files are intentionally not redistributed; obtain them from the links documented in the paper and comply with their respective licenses.

## Outputs

Each run writes `metrics.csv`, `predictions.csv`, `run.json`, and `audit-ledger.jsonl` beneath the output directory. The ledger is a reproducibility aid, not a cryptocurrency or distributed consensus network.

## Academic showcase notebooks

The [`notebooks`](notebooks/) directory contains three concise, publication-safe walkthroughs—one per paper dataset. They use relative paths, contain no private machine details, avoid embedded multi-megabyte training logs, and delegate repeated model code to the tested package. See [`data/README.md`](data/README.md) for the expected filenames.

## Reproduce the paper tables

```bash
python -m warehouse_forecasting.paper_results
```

The command writes the metrics transcribed from Tables 2–6 to `artifacts/paper_metrics.csv` and reports the best model for each dataset/metric. Values are preserved as reported; inconsistencies between the abstract and result tables are not silently corrected.

## License

MIT © 2025 Mohd Ashraf Huzairie
