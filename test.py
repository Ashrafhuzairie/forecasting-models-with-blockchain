import tempfile
import unittest
from pathlib import Path

from warehouse_forecasting.data import demo_series, supervised_windows
from warehouse_forecasting.ledger import append_record, verify
from warehouse_forecasting.metrics import regression_metrics
from warehouse_forecasting.paper_results import table


class ForecastingTests(unittest.TestCase):
    def test_windows(self):
        x, y, dates = supervised_windows(demo_series(50), 10)
        self.assertEqual(x.shape, (40, 10))
        self.assertEqual(len(y), len(dates))

    def test_metrics(self):
        result = regression_metrics([1, 2, 3], [1, 2, 3])
        self.assertEqual(result["rmse"], 0)
        self.assertEqual(result["r2"], 1)

    def test_ledger_detects_no_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "ledger.jsonl"
            append_record(path, {"rmse": 1.2})
            append_record(path, {"rmse": 1.1})
            self.assertTrue(verify(path))

    def test_paper_table_shape(self):
        self.assertEqual(table().shape, (120, 4))


if __name__ == "__main__":
    unittest.main()
