import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import api.realtime_api as realtime_api
import dashboard.factor_lab_api as factor_lab_api
import dashboard.signal_construction_api as signal_construction_api


def _load_module_from_path(name: str, relative_path: str):
    module_path = (REPO_ROOT / relative_path).resolve()
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to build module spec for {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_api_index_module_with_stubs():
    fastapi_module = types.ModuleType("fastapi")
    fastapi_middleware_module = types.ModuleType("fastapi.middleware")
    fastapi_cors_module = types.ModuleType("fastapi.middleware.cors")
    fastapi_responses_module = types.ModuleType("fastapi.responses")

    class FakeFastAPI:
        def __init__(self, *args, **kwargs):
            pass

        def add_middleware(self, *args, **kwargs):
            return None

        def get(self, *args, **kwargs):
            def decorator(fn):
                return fn
            return decorator

        def post(self, *args, **kwargs):
            def decorator(fn):
                return fn
            return decorator

    class FakeRequest:
        async def json(self):
            return {}

    class FakeJSONResponse:
        def __init__(self, content, status_code=200):
            self.content = content
            self.status_code = status_code

    fastapi_module.FastAPI = FakeFastAPI
    fastapi_module.Request = FakeRequest
    fastapi_cors_module.CORSMiddleware = object
    fastapi_responses_module.JSONResponse = FakeJSONResponse

    module_name = "api_index_stubbed_for_tests"
    module_path = (REPO_ROOT / "api" / "index.py").resolve()
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load api/index.py")
    mod = importlib.util.module_from_spec(spec)

    patchers = [
        patch.dict(sys.modules, {
            "fastapi": fastapi_module,
            "fastapi.middleware": fastapi_middleware_module,
            "fastapi.middleware.cors": fastapi_cors_module,
            "fastapi.responses": fastapi_responses_module,
        }),
    ]
    for patcher in patchers:
        patcher.start()
    try:
        spec.loader.exec_module(mod)
    finally:
        for patcher in reversed(patchers):
            patcher.stop()
    return mod


class ApiEntrypointImportTests(unittest.TestCase):
    def test_dashboard_modules_load_via_file_loader(self) -> None:
        modules = {
            "factor_lab_api": "dashboard/factor_lab_api.py",
            "signal_construction_api": "dashboard/signal_construction_api.py",
            "stock_filter_api": "dashboard/stock_filter_api.py",
        }
        for name, rel_path in modules.items():
            with self.subTest(module=name):
                mod = _load_module_from_path(name, rel_path)
                self.assertTrue(hasattr(mod, "_main"))

    def test_api_index_mode_validation_and_heavy_factor_detection(self) -> None:
        api_index = _load_api_index_module_with_stubs()

        self.assertEqual(
            api_index._validate_mode("RUN", {"run", "catalog"}),
            "run",
        )
        with self.assertRaises(ValueError):
            api_index._validate_mode("oops", {"run", "catalog"})

        self.assertTrue(
            api_index._contains_serverless_heavy_factors(
                {"factors": [{"name": "five_factor_axis_value"}]}
            )
        )
        self.assertFalse(
            api_index._contains_serverless_heavy_factors(
                {"factors": [{"name": "momentum"}]}
            )
        )


class NumericGuardTests(unittest.TestCase):
    def test_factor_as_float_rejects_non_finite(self) -> None:
        self.assertEqual(factor_lab_api._as_float(float("inf"), 1.0), 1.0)
        self.assertEqual(factor_lab_api._as_float(float("-inf"), 1.0), 1.0)
        self.assertEqual(factor_lab_api._as_float(float("nan"), 1.0), 1.0)

    def test_factor_normalize_weights_sanitizes_non_finite(self) -> None:
        entries = [
            {"name": "a", "weight": float("inf")},
            {"name": "b", "weight": float("nan")},
            {"name": "c", "weight": -3.0},
            {"name": "d", "weight": 2.0},
        ]
        normalized = factor_lab_api._normalize_weights(entries)
        by_name = {row["name"]: row["weight"] for row in normalized}
        self.assertEqual(by_name["a"], 0.0)
        self.assertEqual(by_name["b"], 0.0)
        self.assertEqual(by_name["c"], 0.0)
        self.assertEqual(by_name["d"], 1.0)

    def test_signal_as_float_rejects_non_finite(self) -> None:
        self.assertEqual(signal_construction_api._as_float(float("inf"), 0.2), 0.2)
        self.assertEqual(signal_construction_api._as_float(float("nan"), 0.2), 0.2)

    def test_signal_runtime_config_rejects_invalid_threshold_order(self) -> None:
        with self.assertRaises(ValueError):
            signal_construction_api._resolve_runtime_config(
                {"buy_threshold": -0.2, "sell_threshold": 0.2}
            )

    def test_realtime_to_float_rejects_non_finite(self) -> None:
        self.assertIsNone(realtime_api._to_float(float("inf")))
        self.assertIsNone(realtime_api._to_float(float("-inf")))
        self.assertIsNone(realtime_api._to_float(float("nan")))


class RealtimePortfolioMathTests(unittest.TestCase):
    def test_portfolio_totals_only_use_cost_covered_positions(self) -> None:
        quote_map = {
            "THYAO": {"last_price": 120.0, "change_pct": 1.2},
            "AKBNK": {"last_price": 50.0, "change_pct": -0.5},
        }

        def fake_quote(symbol: str):
            return quote_map[symbol]

        payload = {
            "holdings": {"THYAO": 10, "AKBNK": 5},
            "cost_basis": {"THYAO": 100},
        }

        with patch.object(realtime_api, "_quote_payload", side_effect=fake_quote):
            result = realtime_api.get_portfolio(payload)

        # Total market value still reflects all priced holdings.
        self.assertEqual(result["total_value"], 1450.0)

        # PnL metrics are computed only on positions that have a cost basis.
        self.assertEqual(result["total_cost"], 1000.0)
        self.assertEqual(result["total_pnl"], 200.0)
        self.assertEqual(result["total_pnl_pct"], 20.0)
        self.assertEqual(result["priced_with_cost_basis"], 1)


if __name__ == "__main__":
    unittest.main()
