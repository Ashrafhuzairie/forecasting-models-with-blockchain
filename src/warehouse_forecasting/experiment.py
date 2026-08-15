from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.arima.model import ARIMA

from .data import supervised_windows
from .ledger import append_record
from .metrics import regression_metrics
from .models import RBFRegressor, build_model


def _fit_predict(name, x_train, y_train, x_test, y_test, lookback, seed, epochs, batch_size):
    if name in {"arima", "arima-rbf"}:
        history = list(y_train)
        predictions = []
        residual_x, residual_y = [], []
        for row, actual in zip(x_test, y_test):
            fit = ARIMA(history, order=(5, 1, 0)).fit()
            base = float(fit.forecast()[0])
            if name == "arima-rbf" and len(residual_y) >= 20:
                correction = RBFRegressor(n_centers=10, random_state=seed).fit(np.asarray(residual_x), residual_y).predict(row[None])[0]
                base += correction
            predictions.append(base)
            history.append(float(actual))
            residual_x.append(row)
            residual_y.append(float(actual - base))
        return np.asarray(predictions)
    model = build_model(name, lookback, seed)
    if name in {"rnn", "lstm", "rnn-lstm", "gan"}:
        model.fit(x_train[..., None], y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        return model.predict(x_test[..., None], verbose=0).reshape(-1)
    model.fit(x_train, y_train)
    return np.asarray(model.predict(x_test)).reshape(-1)


def run_experiment(series: pd.Series, models: list[str], output: str | Path, *, lookback=30, n_splits=5, seed=42, epochs=40, batch_size=32):
    """Evaluate models with expanding-window cross-validation and persist artifacts."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    x, y, dates = supervised_windows(series, lookback)
    splitter = TimeSeriesSplit(n_splits=n_splits)
    metric_rows, prediction_rows = [], []
    for name in models:
        for fold, (train_idx, test_idx) in enumerate(splitter.split(x), start=1):
            scaler = MinMaxScaler().fit(x[train_idx].reshape(-1, 1))
            x_train = scaler.transform(x[train_idx].reshape(-1, 1)).reshape(-1, lookback)
            x_test = scaler.transform(x[test_idx].reshape(-1, 1)).reshape(-1, lookback)
            y_train = scaler.transform(y[train_idx, None]).reshape(-1)
            y_test_scaled = scaler.transform(y[test_idx, None]).reshape(-1)
            predicted_scaled = _fit_predict(name, x_train, y_train, x_test, y_test_scaled, lookback, seed + fold, epochs, batch_size)
            predicted = scaler.inverse_transform(predicted_scaled[:, None]).reshape(-1)
            actual = y[test_idx]
            metric_rows.append({"model": name, "fold": fold, **regression_metrics(actual, predicted)})
            prediction_rows.extend({"date": str(date.date()), "model": name, "fold": fold, "actual": float(a), "predicted": float(p)} for date, a, p in zip(dates[test_idx], actual, predicted))
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    metrics.to_csv(output / "metrics.csv", index=False)
    predictions.to_csv(output / "predictions.csv", index=False)
    summary = metrics.groupby("model")[["mse", "rmse", "mae", "r2", "mape"]].mean().reset_index()
    run = {"author": "Mohd Ashraf Huzairie", "models": models, "lookback": lookback, "n_splits": n_splits, "seed": seed, "summary": summary.to_dict(orient="records")}
    (output / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    append_record(output / "audit-ledger.jsonl", run)
    return summary
