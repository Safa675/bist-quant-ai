#!/usr/bin/env python3
"""
Realtime quote API bridge for Next.js route handlers.

Usage:
    python realtime_api.py quote THYAO
    python realtime_api.py quotes THYAO,AKBNK,GARAN
    python realtime_api.py index XU100
    python realtime_api.py market
    python realtime_api.py portfolio '{"holdings":{"THYAO":100},"cost_basis":{"THYAO":250}}'
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from typing import Any

import borsapy as bp


def _normalize_symbol(symbol: str) -> str:
    return str(symbol or "").strip().upper().split(".")[0]


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        parsed = float(value)
        if not math.isfinite(parsed):
            return None
        return parsed
    except Exception:
        return None


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_change_pct(raw_change_pct: Any, value: float | None, prev_close: float | None) -> float | None:
    computed = None
    if value is not None and prev_close not in (None, 0.0):
        computed = ((value - prev_close) / prev_close) * 100.0

    provided = _to_float(raw_change_pct)
    if provided is None:
        return computed
    if computed is None:
        return provided

    if abs(provided - computed) <= 0.15:
        return provided
    if abs((provided * 100.0) - computed) <= 0.15:
        return provided * 100.0
    return computed


def _quote_payload(symbol: str) -> dict[str, Any]:
    symbol = _normalize_symbol(symbol)
    if not symbol:
        return {"symbol": symbol, "error": "Invalid symbol"}

    try:
        ticker = bp.Ticker(symbol)
        fast_info = dict(ticker.fast_info)

        last_price = _to_float(fast_info.get("last_price"))
        prev_close = _to_float(fast_info.get("previous_close"))
        change = None
        change_pct = None
        if last_price is not None and prev_close not in (None, 0.0):
            change = last_price - prev_close
            change_pct = (change / prev_close) * 100.0

        return {
            "symbol": symbol,
            "last_price": last_price,
            "change": change,
            "change_pct": change_pct,
            "volume": _to_float(fast_info.get("volume")),
            "bid": None,
            "ask": None,
            "high": _to_float(fast_info.get("day_high")),
            "low": _to_float(fast_info.get("day_low")),
            "open": _to_float(fast_info.get("open")),
            "prev_close": prev_close,
            "market_cap": _to_float(fast_info.get("market_cap")),
            "timestamp": _utc_iso_now(),
        }
    except Exception as exc:
        return {"symbol": symbol, "error": str(exc), "timestamp": _utc_iso_now()}


def get_quote(symbol: str) -> dict[str, Any]:
    return _quote_payload(symbol)


def get_quotes(symbols: list[str]) -> dict[str, Any]:
    normalized = [_normalize_symbol(s) for s in symbols]
    normalized = [s for s in normalized if s]
    quotes = {symbol: _quote_payload(symbol) for symbol in normalized}
    return {
        "quotes": quotes,
        "count": len(quotes),
        "timestamp": _utc_iso_now(),
    }


def get_index(index_name: str) -> dict[str, Any]:
    index_name = _normalize_symbol(index_name) or "XU100"
    try:
        idx = bp.index(index_name)
        info = idx.info if isinstance(idx.info, dict) else {}
        value = _to_float(info.get("last"))
        prev_close = _to_float(info.get("close"))
        change_pct = _normalize_change_pct(info.get("change_percent"), value, prev_close)
        return {
            "index": index_name,
            "value": value,
            "change_pct": change_pct,
            "prev_close": prev_close,
            "timestamp": _utc_iso_now(),
        }
    except Exception as exc:
        return {"index": index_name, "error": str(exc), "timestamp": _utc_iso_now()}


def get_portfolio(payload: dict[str, Any]) -> dict[str, Any]:
    holdings_raw = payload.get("holdings", payload)
    cost_basis_raw = payload.get("cost_basis", {})

    if not isinstance(holdings_raw, dict):
        return {"error": "Holdings must be an object"}

    holdings: dict[str, float] = {}
    for symbol, qty in holdings_raw.items():
        norm = _normalize_symbol(symbol)
        if not norm:
            continue
        quantity = _to_float(qty)
        if quantity is None:
            continue
        holdings[norm] = quantity

    cost_basis: dict[str, float] = {}
    if isinstance(cost_basis_raw, dict):
        for symbol, cost in cost_basis_raw.items():
            norm = _normalize_symbol(symbol)
            if not norm:
                continue
            parsed_cost = _to_float(cost)
            if parsed_cost is not None:
                cost_basis[norm] = parsed_cost

    if not holdings:
        return {"error": "No valid holdings provided"}

    quotes = {symbol: _quote_payload(symbol) for symbol in holdings.keys()}
    positions: list[dict[str, Any]] = []
    total_value = 0.0
    total_cost = 0.0
    total_value_with_cost = 0.0
    positions_with_cost_basis = 0

    for symbol, quantity in holdings.items():
        quote = quotes.get(symbol, {})
        price = _to_float(quote.get("last_price"))
        if price is None:
            positions.append(
                {
                    "symbol": symbol,
                    "quantity": quantity,
                    "error": quote.get("error", "No price data"),
                }
            )
            continue

        market_value = price * quantity
        total_value += market_value
        row: dict[str, Any] = {
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            "market_value": market_value,
            "change_pct": quote.get("change_pct"),
        }

        if symbol in cost_basis:
            unit_cost = cost_basis[symbol]
            cost = unit_cost * quantity
            pnl = market_value - cost
            pnl_pct = (pnl / cost * 100.0) if cost > 0 else 0.0
            total_cost += cost
            total_value_with_cost += market_value
            positions_with_cost_basis += 1
            row.update(
                {
                    "cost_basis": unit_cost,
                    "cost": cost,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                }
            )
        positions.append(row)

    result: dict[str, Any] = {
        "total_value": total_value,
        "positions": sorted(positions, key=lambda x: x.get("market_value", 0.0), reverse=True),
        "timestamp": _utc_iso_now(),
    }
    if positions_with_cost_basis > 0:
        total_pnl = total_value_with_cost - total_cost
        result.update(
            {
                "total_cost": total_cost,
                "total_pnl": total_pnl,
                "total_pnl_pct": (total_pnl / total_cost * 100.0) if total_cost > 0 else 0.0,
                "priced_with_cost_basis": positions_with_cost_basis,
            }
        )
    return result


def get_market() -> dict[str, Any]:
    summary: dict[str, Any] = {"timestamp": _utc_iso_now()}
    summary["xu100"] = get_index("XU100")
    summary["xu030"] = get_index("XU030")

    try:
        fx = bp.FX("USD")
        info = fx.info if isinstance(fx.info, dict) else {}
        rate = _to_float(info.get("last"))
        open_rate = _to_float(info.get("open"))
        change_pct = None
        if rate is not None and open_rate not in (None, 0.0):
            change_pct = ((rate - open_rate) / open_rate) * 100.0
        summary["usdtry"] = {"rate": rate, "change_pct": change_pct}
    except Exception as exc:
        summary["usdtry"] = {"error": str(exc)}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime API bridge")
    parser.add_argument("command", choices=["quote", "quotes", "index", "portfolio", "market"])
    parser.add_argument("args", nargs="?", default="")
    args = parser.parse_args()

    if args.command == "quote":
        result = get_quote(args.args)
    elif args.command == "quotes":
        symbols = [s.strip() for s in args.args.split(",") if s.strip()]
        result = get_quotes(symbols)
    elif args.command == "index":
        result = get_index(args.args or "XU100")
    elif args.command == "market":
        result = get_market()
    elif args.command == "portfolio":
        try:
            payload = json.loads(args.args) if args.args else {}
            if not isinstance(payload, dict):
                payload = {}
        except json.JSONDecodeError:
            payload = {}
        result = get_portfolio(payload)
    else:
        result = {"error": f"Unsupported command: {args.command}"}

    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
