"""
Unified BIST Engine API for Vercel

Combines Factor Lab and Signal Construction into a single Python serverless function.
This is the main entry point for all Python-based API calls.
"""

from __future__ import annotations

import importlib.util
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from dashboard.common_response import error_response, generate_run_id, success_response

# Setup paths
ROOT = Path(__file__).resolve().parent.parent
FACTOR_MODULE_PATH = ROOT / "dashboard" / "factor_lab_api.py"
SIGNAL_MODULE_PATH = ROOT / "dashboard" / "signal_construction_api.py"
STOCK_FILTER_MODULE_PATH = ROOT / "dashboard" / "stock_filter_api.py"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module(name: str, path: Path):
    """Dynamically load a Python module from file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Lazy load modules to reduce cold start time
_factor_lab_api = None
_signal_construction_api = None
_stock_filter_api = None


def get_factor_lab_api():
    global _factor_lab_api
    if _factor_lab_api is None:
        _factor_lab_api = _load_module("factor_lab_api", FACTOR_MODULE_PATH)
    return _factor_lab_api


def get_signal_construction_api():
    global _signal_construction_api
    if _signal_construction_api is None:
        _signal_construction_api = _load_module("signal_construction_api", SIGNAL_MODULE_PATH)
    return _signal_construction_api


def get_stock_filter_api():
    global _stock_filter_api
    if _stock_filter_api is None:
        _stock_filter_api = _load_module("stock_filter_api", STOCK_FILTER_MODULE_PATH)
    return _stock_filter_api


# Create FastAPI app
app = FastAPI(title="BIST Quant Engine", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _error(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status_code)


def _error_envelope(
    message: str,
    *,
    run_id: str,
    engine: str,
    mode: str | None = None,
    status_code: int = 400,
) -> JSONResponse:
    meta = {"engine": engine}
    if mode is not None:
        meta["mode"] = mode
    return JSONResponse(
        error_response(message, run_id=run_id, meta=meta),
        status_code=status_code,
    )


def _resolve_run_id(payload: Any, prefix: str) -> str:
    candidate = payload.get("run_id") if isinstance(payload, dict) else None
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()
    return generate_run_id(prefix)


def _validate_mode(mode: str, allowed: set[str]) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in allowed:
        allowed_text = ", ".join(sorted(allowed))
        raise ValueError(f"Unsupported mode '{normalized}'. Use one of: {allowed_text}.")
    return normalized


def _contains_serverless_heavy_factors(payload: dict[str, Any]) -> bool:
    factor_key = str(payload.get("factor_key", "")).strip().lower()
    if factor_key == "five_factor_rotation" or factor_key.startswith("five_factor_axis_"):
        return True

    raw_factors = payload.get("factors")
    if not isinstance(raw_factors, list):
        return False

    for item in raw_factors:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip().lower()
        if name == "five_factor_rotation" or name.startswith("five_factor_axis_"):
            return True
    return False


def _factor_catalog() -> dict[str, Any]:
    """Build factor catalog from the engine."""
    factor_lab_api = get_factor_lab_api()
    with redirect_stdout(io.StringIO()):
        engine = factor_lab_api.PortfolioEngine(
            data_dir=ROOT / "data",
            regime_model_dir=ROOT / "Regime Filter",
            start_date="2014-01-01",
            end_date="2026-12-31",
        )

    return {
        "factors": factor_lab_api._build_factor_catalog(engine),
        "default_portfolio_options": factor_lab_api.DEFAULT_PORTFOLIO_OPTIONS,
    }


# ====================
# Health & Status
# ====================

@app.get("/")
async def root() -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "service": "bist-quant-engine",
        "version": "1.0.0",
        "endpoints": [
            "/api/health",
            "/api/factor_lab",
            "/api/signal_construction",
            "/api/stock_filter",
        ]
    })


@app.get("/api/health")
@app.get("/py-api/api/health")
async def health() -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "service": "bist-quant-engine",
        "python_version": sys.version,
    })


# ====================
# Factor Lab API
# ====================

@app.get("/api/factor_lab")
@app.get("/py-api/api/factor_lab")
async def get_factor_catalog() -> JSONResponse:
    """Get available factors and portfolio options."""
    run_id = generate_run_id("factor_lab")
    try:
        return JSONResponse(success_response(
            _factor_catalog(),
            run_id=run_id,
            meta={"engine": "factor_lab", "mode": "catalog"},
        ))
    except Exception as exc:
        return _error_envelope(
            str(exc),
            run_id=run_id,
            engine="factor_lab",
            mode="catalog",
            status_code=500,
        )


@app.post("/api/factor_lab")
@app.post("/py-api/api/factor_lab")
async def post_factor_run(request: Request) -> JSONResponse:
    """Run factor analysis or backtest."""
    try:
        payload: Any = await request.json()
    except Exception:
        return _error("Request body must be valid JSON.")

    if not isinstance(payload, dict):
        return _error("Request body must be a JSON object.")

    run_id = _resolve_run_id(payload, "factor_lab")
    try:
        mode = _validate_mode(payload.get("_mode", "run"), {"run", "catalog"})
    except ValueError as exc:
        return _error_envelope(
            str(exc),
            run_id=run_id,
            engine="factor_lab",
            mode=str(payload.get("_mode", "run")),
            status_code=400,
        )

    # Block heavy computation on Vercel
    if _contains_serverless_heavy_factors(payload):
        return _error_envelope(
            "five_factor_rotation is too computationally intensive for serverless. "
            "Use a different factor or deploy to a dedicated server.",
            run_id=run_id,
            engine="factor_lab",
            mode=mode,
            status_code=400,
        )

    try:
        factor_lab_api = get_factor_lab_api()
        if mode == "catalog":
            return JSONResponse(success_response(
                _factor_catalog(),
                run_id=run_id,
                meta={"engine": "factor_lab", "mode": mode},
            ))

        with redirect_stdout(io.StringIO()):
            response = factor_lab_api._build_response(payload)
        return JSONResponse(success_response(
            response,
            run_id=run_id,
            meta={"engine": "factor_lab", "mode": mode},
        ))
    except ValueError as exc:
        return _error_envelope(
            str(exc),
            run_id=run_id,
            engine="factor_lab",
            mode=mode,
            status_code=400,
        )
    except Exception as exc:
        return _error_envelope(
            str(exc),
            run_id=run_id,
            engine="factor_lab",
            mode=mode,
            status_code=500,
        )


# ====================
# Signal Construction API
# ====================

@app.get("/api/signal_construction")
@app.get("/py-api/api/signal_construction")
async def get_signal_metadata() -> JSONResponse:
    """Get available indicators and options for signal construction."""
    run_id = generate_run_id("signal_construction")
    return JSONResponse(success_response(
        {
            "universes": ["XU030", "XU100", "XUTUM", "CUSTOM"],
            "periods": ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            "intervals": ["1d"],
            "indicators": [
                {"key": "rsi", "label": "RSI"},
                {"key": "macd", "label": "MACD Histogram"},
                {"key": "bollinger", "label": "Bollinger %B"},
                {"key": "atr", "label": "ATR (Cross-Sectional)"},
                {"key": "stochastic", "label": "Stochastic %K"},
                {"key": "adx", "label": "ADX (+DI/-DI trend)"},
                {"key": "supertrend", "label": "Supertrend Direction"},
            ],
        },
        run_id=run_id,
        meta={"engine": "signal_construction", "mode": "meta"},
    ))


@app.post("/api/signal_construction")
@app.post("/py-api/api/signal_construction")
async def post_signal_run(request: Request) -> JSONResponse:
    """Construct signals or run backtest."""
    try:
        payload: Any = await request.json()
    except Exception:
        return _error("Request body must be valid JSON.")

    if not isinstance(payload, dict):
        return _error("Request body must be a JSON object.")

    run_id = _resolve_run_id(payload, "signal_construction")
    try:
        mode = _validate_mode(payload.get("_mode", "construct"), {"construct", "backtest"})
    except ValueError as exc:
        return _error_envelope(
            str(exc),
            run_id=run_id,
            engine="signal_construction",
            mode=str(payload.get("_mode", "construct")),
            status_code=400,
        )

    try:
        signal_construction_api = get_signal_construction_api()
        if mode == "backtest":
            with redirect_stdout(io.StringIO()):
                response = signal_construction_api._build_backtest_response(payload)
        else:
            with redirect_stdout(io.StringIO()):
                response = signal_construction_api._build_response(payload)

        return JSONResponse(success_response(
            response,
            run_id=run_id,
            meta={"engine": "signal_construction", "mode": mode},
        ))
    except ValueError as exc:
        return _error_envelope(
            str(exc),
            run_id=run_id,
            engine="signal_construction",
            mode=mode,
            status_code=400,
        )
    except Exception as exc:
        return _error_envelope(
            str(exc),
            run_id=run_id,
            engine="signal_construction",
            mode=mode,
            status_code=500,
        )


# ====================
# Stock Filter API
# ====================

@app.get("/api/stock_filter")
@app.get("/py-api/api/stock_filter")
async def get_stock_filter_meta() -> JSONResponse:
    """Get templates/filter fields for stock screening."""
    run_id = generate_run_id("stock_filter")
    try:
        stock_filter_api = get_stock_filter_api()
        with redirect_stdout(io.StringIO()):
            response = stock_filter_api._meta_response()
        return JSONResponse(success_response(
            response,
            run_id=run_id,
            meta={"engine": "stock_filter", "mode": "meta"},
        ))
    except Exception as exc:
        return _error_envelope(
            str(exc),
            run_id=run_id,
            engine="stock_filter",
            mode="meta",
            status_code=500,
        )


@app.post("/api/stock_filter")
@app.post("/py-api/api/stock_filter")
async def post_stock_filter_run(request: Request) -> JSONResponse:
    """Run stock screener with filters."""
    try:
        payload: Any = await request.json()
    except Exception:
        return _error("Request body must be valid JSON.")

    if not isinstance(payload, dict):
        return _error("Request body must be a JSON object.")

    run_id = _resolve_run_id(payload, "stock_filter")
    try:
        mode = _validate_mode(payload.get("_mode", "run"), {"run", "meta"})
    except ValueError as exc:
        return _error_envelope(
            str(exc),
            run_id=run_id,
            engine="stock_filter",
            mode=str(payload.get("_mode", "run")),
            status_code=400,
        )

    try:
        stock_filter_api = get_stock_filter_api()
        if mode == "meta":
            with redirect_stdout(io.StringIO()):
                response = stock_filter_api._meta_response()
        else:
            with redirect_stdout(io.StringIO()):
                response = stock_filter_api._run_response(payload)
        return JSONResponse(success_response(
            response,
            run_id=run_id,
            meta={"engine": "stock_filter", "mode": mode},
        ))
    except ValueError as exc:
        return _error_envelope(
            str(exc),
            run_id=run_id,
            engine="stock_filter",
            mode=mode,
            status_code=400,
        )
    except Exception as exc:
        return _error_envelope(
            str(exc),
            run_id=run_id,
            engine="stock_filter",
            mode=mode,
            status_code=500,
        )
