# Academic showcase notebooks

Author: **Mohd Ashraf Huzairie**

These notebooks are intentionally concise versions of the original experiments. They share one reproducible structure and import preprocessing, validation, models, and metrics from `warehouse_forecasting`.

1. `01_dataset1_retail_supply_chain.ipynb`
2. `02_dataset2_historical_product_demand.ipynb`
3. `03_dataset3_demand_forecasting_kernels.ipynb`

Install the repository before opening Jupyter:

```bash
pip install -e .
jupyter lab
```

The default notebook mode performs data validation and EDA without starting expensive training. Set `RUN_TRAINING = True` in a notebook to run the lightweight MLP and RBF comparison. Use the command-line interface for the complete eight-model study.
