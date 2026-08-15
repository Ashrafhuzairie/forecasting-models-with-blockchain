# Dataset setup

The source datasets are not committed because they are third-party Kaggle assets. Download them from their respective publishers and use these filenames:

| Paper dataset | Expected local path |
|---|---|
| Dataset 1 — Retail Supply Chain Sales | `data/dataset1/Retail-Supply-Chain-Sales-Dataset.xlsx` |
| Dataset 2 — Historical Product Demand | `data/dataset2/Historical Product Demand.csv` |
| Dataset 3 — Demand Forecasting Kernels | `data/dataset3/train.csv` |

Dataset 3's `test.csv` does not contain true sales values and is therefore not used for evaluation. Do not replace its missing targets with zero and report those values as measured forecast accuracy.

Data source links are documented in the IEEE paper and project README. Review and comply with the Kaggle dataset licenses before redistributing any data.
