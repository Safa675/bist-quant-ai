import unittest

from dashboard.analytics_payloads import _turnover_metrics


class TurnoverMetricsTests(unittest.TestCase):
    def test_normalizes_ticker_suffixes(self) -> None:
        holdings_history = [
            {"date": "2024-01-01", "ticker": "THYAO.IS"},
            {"date": "2024-01-02", "ticker": "THYAO"},
        ]
        metrics = _turnover_metrics(holdings_history)
        self.assertEqual(metrics["rebalance_events"], 1)
        self.assertEqual(metrics["avg_positions"], 1.0)
        self.assertEqual(metrics["avg_turnover"], 0.0)

    def test_full_replacement_turnover_is_one(self) -> None:
        holdings_history = [
            {"date": "2024-01-01", "ticker": "A"},
            {"date": "2024-01-01", "ticker": "B"},
            {"date": "2024-01-02", "ticker": "C"},
            {"date": "2024-01-02", "ticker": "D"},
        ]
        metrics = _turnover_metrics(holdings_history)
        self.assertEqual(metrics["rebalance_events"], 1)
        self.assertEqual(metrics["avg_positions"], 2.0)
        self.assertEqual(metrics["avg_turnover"], 1.0)

    def test_uses_weight_based_one_way_turnover(self) -> None:
        holdings_history = [
            {"date": "2024-01-01", "ticker": "A", "weight": 0.6},
            {"date": "2024-01-01", "ticker": "B", "weight": 0.4},
            {"date": "2024-01-02", "ticker": "A", "weight": 0.3},
            {"date": "2024-01-02", "ticker": "B", "weight": 0.3},
            {"date": "2024-01-02", "ticker": "C", "weight": 0.4},
        ]
        metrics = _turnover_metrics(holdings_history)
        self.assertEqual(metrics["rebalance_events"], 1)
        self.assertAlmostEqual(metrics["avg_turnover"], 0.4, places=8)


if __name__ == "__main__":
    unittest.main()
