import re
import unittest
from unittest.mock import patch

from dashboard.common_response import error_response, success_response
import api.realtime_api as realtime_api


class CommonResponseTests(unittest.TestCase):
    def test_success_response_preserves_server_timestamp(self) -> None:
        payload = success_response(
            {"ok": True},
            run_id="  run_123  ",
            meta={"engine": "x", "timestamp": "spoofed"},
        )
        self.assertEqual(payload["run_id"], "run_123")
        self.assertEqual(payload["meta"]["engine"], "x")
        self.assertNotEqual(payload["meta"]["timestamp"], "spoofed")
        self.assertRegex(payload["meta"]["timestamp"], r"\+00:00$")

    def test_error_response_coerces_non_string_run_id(self) -> None:
        payload = error_response("boom", run_id=123)
        self.assertEqual(payload["run_id"], "123")
        self.assertRegex(payload["meta"]["timestamp"], r"\+00:00$")

    def test_blank_run_id_falls_back_to_generated(self) -> None:
        payload = success_response({"ok": True}, run_id="   ")
        self.assertTrue(payload["run_id"].startswith("run_"))
        self.assertGreater(len(payload["run_id"]), 4)


class RealtimeApiTests(unittest.TestCase):
    def test_index_change_pct_normalizes_decimal_fraction(self) -> None:
        class FakeIndex:
            info = {"last": 101.5, "close": 100.0, "change_percent": 0.015}

        with patch.object(realtime_api.bp, "index", return_value=FakeIndex()):
            result = realtime_api.get_index("XU100")

        self.assertAlmostEqual(result["change_pct"], 1.5, places=6)
        self.assertRegex(result["timestamp"], r"\+00:00$")

    def test_index_change_pct_keeps_percent_units(self) -> None:
        class FakeIndex:
            info = {"last": 101.5, "close": 100.0, "change_percent": 1.5}

        with patch.object(realtime_api.bp, "index", return_value=FakeIndex()):
            result = realtime_api.get_index("XU100")

        self.assertAlmostEqual(result["change_pct"], 1.5, places=6)

    def test_quote_timestamp_is_utc(self) -> None:
        class FakeTicker:
            fast_info = {
                "last_price": 10.0,
                "previous_close": 9.5,
                "volume": 1000,
                "day_high": 10.2,
                "day_low": 9.7,
                "open": 9.8,
                "market_cap": 1_000_000,
            }

        with patch.object(realtime_api.bp, "Ticker", return_value=FakeTicker()):
            quote = realtime_api.get_quote("THYAO")

        self.assertEqual(quote["symbol"], "THYAO")
        self.assertRegex(quote["timestamp"], r"\+00:00$")


if __name__ == "__main__":
    unittest.main()
