#!/usr/bin/env python3
"""
Stock Filter API for Quant AI.

Reads JSON from stdin and writes JSON to stdout.
Modes:
- meta: return available templates/filter fields
- run: execute stock screening with fundamental filters
"""

from __future__ import annotations

import io
import json
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

import borsapy as bp  # noqa: E402


FILTER_FIELD_DEFS: list[dict[str, Any]] = [
    {"key": "market_cap_usd", "label": "Market Cap (USD mn)", "group": "valuation"},
    {"key": "market_cap", "label": "Market Cap (TL mn)", "group": "valuation"},
    {"key": "pe", "label": "P/E", "group": "valuation"},
    {"key": "pb", "label": "P/B", "group": "valuation"},
    {"key": "ev_ebitda", "label": "EV/EBITDA", "group": "valuation"},
    {"key": "ev_sales", "label": "EV/Sales", "group": "valuation"},
    {"key": "dividend_yield", "label": "Dividend Yield (%)", "group": "income"},
    {"key": "upside_potential", "label": "Upside Potential (%)", "group": "analyst"},
    {"key": "roe", "label": "ROE (%)", "group": "quality"},
    {"key": "roa", "label": "ROA (%)", "group": "quality"},
    {"key": "net_margin", "label": "Net Margin (%)", "group": "quality"},
    {"key": "ebitda_margin", "label": "EBITDA Margin (%)", "group": "quality"},
    {"key": "foreign_ratio", "label": "Foreign Ownership (%)", "group": "flow"},
    {"key": "float_ratio", "label": "Free Float (%)", "group": "flow"},
    {"key": "volume_3m", "label": "Avg Volume 3M (mn)", "group": "liquidity"},
    {"key": "volume_12m", "label": "Avg Volume 12M (mn)", "group": "liquidity"},
    {"key": "return_1w", "label": "Return 1W (%)", "group": "momentum"},
    {"key": "return_1m", "label": "Return 1M (%)", "group": "momentum"},
    {"key": "return_1y", "label": "Return 1Y (%)", "group": "momentum"},
    {"key": "return_ytd", "label": "Return YTD (%)", "group": "momentum"},
]

FIELD_LABELS = {row["key"]: row["label"] for row in FILTER_FIELD_DEFS}

DISPLAY_COLUMNS_DEFAULT: list[str] = [
    "symbol",
    "name",
    "market_cap_usd",
    "pe",
    "pb",
    "dividend_yield",
    "upside_potential",
    "roe",
    "net_margin",
    "return_1m",
]

INDEX_OPTIONS = ["XU030", "XU050", "XU100", "XUTUM"]
RECOMMENDATION_OPTIONS = ["AL", "TUT", "SAT"]


def _parse_payload() -> dict[str, Any]:
    raw = sys.stdin.read().strip()
    if not raw:
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")
    return payload


def _as_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    return max(minimum, min(maximum, parsed))


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
        if np.isnan(parsed) or np.isinf(parsed):
            return None
        return parsed
    except Exception:
        return None


def _safe_bool(value: Any, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if pd.isna(value):
        return None
    return str(value)


def _normalize_filters(raw: Any, allowed: set[str]) -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {}
    if not isinstance(raw, dict):
        return out

    for key, value in raw.items():
        name = str(key).strip()
        if not name or name not in allowed:
            continue

        if not isinstance(value, dict):
            continue

        minimum = _as_float(value.get("min"))
        maximum = _as_float(value.get("max"))
        if minimum is None and maximum is None:
            continue

        if minimum is not None and maximum is not None and minimum > maximum:
            minimum, maximum = maximum, minimum

        out[name] = {"min": minimum, "max": maximum}

    return out


def _friendly_error(exc: Exception) -> str:
    message = str(exc)
    if "CERTIFICATE_VERIFY_FAILED" in message:
        return (
            "Stock screener SSL verification failed in this runtime. "
            "Use deployment runtime or install CA certificates on this host."
        )
    return message


def _meta_response() -> dict[str, Any]:
    templates = sorted(getattr(bp.Screener, "TEMPLATES", []))
    return {
        "templates": templates,
        "filters": FILTER_FIELD_DEFS,
        "indexes": INDEX_OPTIONS,
        "recommendations": RECOMMENDATION_OPTIONS,
        "default_sort_by": "upside_potential",
        "default_sort_desc": True,
    }


def _run_response(payload: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()

    screener = bp.Screener()
    provider = getattr(screener, "_provider", None)
    criteria_map_raw = getattr(provider, "CRITERIA_MAP", {}) if provider is not None else {}
    criteria_map = {
        str(k): str(v)
        for k, v in criteria_map_raw.items()
        if isinstance(k, str)
    }
    reverse_criteria_map = {value: key for key, value in criteria_map.items()}

    allowed_filters = set(criteria_map.keys()) or set(FIELD_LABELS.keys())

    template = str(payload.get("template") or "").strip()
    if template and template not in set(getattr(bp.Screener, "TEMPLATES", [])):
        raise ValueError(f"Unknown template: {template}")

    sector = str(payload.get("sector") or "").strip()
    index_name = str(payload.get("index") or "").strip().upper()
    recommendation = str(payload.get("recommendation") or "").strip().upper()

    if sector:
        screener.set_sector(sector)
    if index_name:
        screener.set_index(index_name)
    if recommendation:
        screener.set_recommendation(recommendation)

    filters = _normalize_filters(payload.get("filters"), allowed_filters)
    applied_filters: list[dict[str, Any]] = []

    for key, bounds in filters.items():
        minimum = bounds.get("min")
        maximum = bounds.get("max")
        screener.add_filter(key, min=minimum, max=maximum)
        applied_filters.append(
            {
                "key": key,
                "label": FIELD_LABELS.get(key, key),
                "min": minimum,
                "max": maximum,
            }
        )

    with redirect_stdout(io.StringIO()):
        df = screener.run(template=template or None)

    if not isinstance(df, pd.DataFrame):
        raise RuntimeError("Screener returned an invalid response.")

    rename_map: dict[str, str] = {}
    for col in df.columns:
        if not isinstance(col, str):
            continue
        if not col.startswith("criteria_"):
            continue
        criterion_id = col.removeprefix("criteria_")
        criterion_name = reverse_criteria_map.get(criterion_id)
        if criterion_name:
            rename_map[col] = criterion_name

    if rename_map:
        df = df.rename(columns=rename_map)

    sort_by = str(payload.get("sort_by") or "upside_potential").strip()
    sort_desc = _safe_bool(payload.get("sort_desc"), default=True)

    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=not sort_desc, na_position="last")

    total_matches = int(len(df))
    limit = _as_int(payload.get("limit"), default=100, minimum=1, maximum=500)
    if total_matches > limit:
        df = df.head(limit)

    requested_columns_raw = payload.get("columns")
    requested_columns: list[str] = []
    if isinstance(requested_columns_raw, list):
        requested_columns = [str(c) for c in requested_columns_raw if isinstance(c, str)]

    if requested_columns:
        display_columns = [col for col in requested_columns if col in df.columns]
    else:
        display_columns = [col for col in DISPLAY_COLUMNS_DEFAULT if col in df.columns]
        for key in filters.keys():
            if key in df.columns and key not in display_columns:
                display_columns.append(key)

    if not display_columns:
        display_columns = [str(c) for c in df.columns[:20]]

    if display_columns:
        out_df = df.loc[:, display_columns].copy()
    else:
        out_df = df.copy()

    rows: list[dict[str, Any]] = []
    for _, row in out_df.iterrows():
        rows.append({col: _jsonable(row[col]) for col in out_df.columns})

    elapsed_ms = int((time.perf_counter() - started) * 1000)

    return {
        "meta": {
            "as_of": pd.Timestamp.utcnow().isoformat(),
            "execution_ms": elapsed_ms,
            "total_matches": total_matches,
            "returned_rows": len(rows),
            "template": template or None,
            "sector": sector or None,
            "index": index_name or None,
            "recommendation": recommendation or None,
            "sort_by": sort_by,
            "sort_desc": sort_desc,
        },
        "columns": [
            {"key": col, "label": FIELD_LABELS.get(col, col.replace("_", " ").title())}
            for col in out_df.columns
        ],
        "rows": rows,
        "applied_filters": applied_filters,
    }


def _main() -> int:
    try:
        payload = _parse_payload()
        mode = str(payload.get("_mode") or "run").strip().lower()

        if mode == "meta":
            response = _meta_response()
        else:
            response = _run_response(payload)

        print(json.dumps(response, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(json.dumps({"error": _friendly_error(exc)}, ensure_ascii=False))
        return 0


if __name__ == "__main__":
    raise SystemExit(_main())
