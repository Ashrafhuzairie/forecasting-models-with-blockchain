from __future__ import annotations

from pathlib import Path

import pandas as pd


MODELS = ["RNN", "LSTM", "RNN-LSTM", "GAN", "MLP", "RBF", "ARIMA", "ARIMA-RBF"]
VALUES = {
    "mse": [[95925.93, 669398073.95, 53.63], [24126.95, 661911970.95, 47.38], [32580.57, 666800846.81, 60.21], [439165.86, 862178506.10, 888.86], [7519.22, 658895823.01, 55.32], [208825.63, 1578964671.82, 79.14], [391139.44, 929513842.60, 884.47], [213727.38, 1780603711.13, 93.77]],
    "rmse": [[303.56, 25768.12, 7.32], [88.83, 25641.65, 6.88], [111.05, 25731.82, 7.76], [644.53, 29301.87, 29.81], [68.67, 25571.15, 7.44], [405.17, 34664.33, 8.90], [600.73, 24031.87, 29.39], [434.91, 38498.06, 9.68]],
    "mae": [[126.84, 5283.15, 5.60], [22.31, 5271.88, 5.29], [39.47, 5684.28, 5.94], [226.17, 5241.30, 24.86], [29.95, 5439.18, 5.68], [60.81, 5278.21, 6.73], [268.76, 6572.37, 23.93], [61.04, 5961.80, 7.42]],
    "r2": [[69, 21, 94], [97, 22, 94], [95, 21, 93], [-16, -2, -7], [99, 22, 93], [57, -90, 90], [0, -107, -24], [51, -106, 89]],
    "mape": [[286.95, 16334.90, 13.19], [62.22, 8629.33, 12.46], [101.46, 16967.98, 14.26], [93.70, 10786.30, 81.91], [109.52, 9960, 13.28], [95.40, 7395.96, 16.07], [1109.04, 37261.21, 70.94], [91.92, 12222.36, 17.84]],
}


def table():
    rows = []
    for metric, model_values in VALUES.items():
        for model, values in zip(MODELS, model_values):
            rows.extend({"model": model, "dataset": f"dataset_{number}", "metric": metric, "value": value} for number, value in enumerate(values, 1))
    return pd.DataFrame(rows)


def main():
    output = Path("artifacts/paper_metrics.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    data = table()
    data.to_csv(output, index=False)
    errors = data[data.metric != "r2"]
    scores = data[data.metric == "r2"]
    best = pd.concat([
        errors.loc[errors.groupby(["dataset", "metric"])["value"].idxmin()],
        scores.loc[scores.groupby(["dataset", "metric"])["value"].idxmax()],
    ]).sort_values(["dataset", "metric"])
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
