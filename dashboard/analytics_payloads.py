"""Detailed analytics payload builders for backtest API responses."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _normalize_ticker(value: Any) -> str:
    raw = str(value or "").strip().upper()
    if not raw:
        return ""
    return raw.split(".")[0]


def _to_series(values: pd.Series | None) -> pd.Series:
    if values is None:
        return pd.Series(dtype="float64")
    series = pd.to_numeric(values, errors="coerce").dropna()
    if series.empty:
        return pd.Series(dtype="float64")
    series = series.astype("float64")
    if isinstance(series.index, pd.DatetimeIndex):
        series = series.sort_index()
    return series


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except Exception:
        return None
    if np.isnan(parsed) or np.isinf(parsed):
        return None
    return parsed


def _yearly_metrics(returns: pd.Series) -> list[dict[str, Any]]:
    if returns.empty or not isinstance(returns.index, pd.DatetimeIndex):
        return []

    rows: list[dict[str, Any]] = []
    for year, group in returns.groupby(returns.index.year):
        if group.empty:
            continue

        equity = (1.0 + group).cumprod()
        total_return = equity.iloc[-1] - 1.0
        std = group.std()
        vol = std * np.sqrt(252.0) if std > 0 else np.nan
        sharpe = (group.mean() / std) * np.sqrt(252.0) if std > 0 else np.nan
        drawdown = equity / equity.cummax() - 1.0
        max_dd = drawdown.min() if not drawdown.empty else np.nan
        win_rate = (group > 0).mean() if len(group) > 0 else np.nan

        rows.append(
            {
                "year": int(year),
                "return": _safe_float(total_return * 100.0),
                "volatility": _safe_float(vol * 100.0),
                "sharpe": _safe_float(sharpe),
                "max_dd": _safe_float(max_dd * 100.0),
                "win_rate": _safe_float(win_rate * 100.0),
                "trading_days": int(len(group)),
            }
        )

    return rows


def _monthly_returns(returns: pd.Series) -> list[dict[str, Any]]:
    if returns.empty or not isinstance(returns.index, pd.DatetimeIndex):
        return []

    monthly = (1.0 + returns).resample("ME").prod() - 1.0
    rows: list[dict[str, Any]] = []
    for dt, value in monthly.items():
        rows.append(
            {
                "year": int(dt.year),
                "month": int(dt.month),
                "return": _safe_float(value * 100.0),
            }
        )
    return rows


def _drawdown_metrics(equity: pd.Series) -> dict[str, Any]:
    if equity.empty:
        return {
            "current_dd": None,
            "max_dd": None,
            "avg_dd": None,
            "max_duration_days": 0,
        }

    dd = equity / equity.cummax() - 1.0
    negative = dd[dd < 0]

    max_streak = 0
    streak = 0
    for value in dd:
        if value < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    return {
        "current_dd": _safe_float(dd.iloc[-1] * 100.0),
        "max_dd": _safe_float(dd.min() * 100.0),
        "avg_dd": _safe_float(negative.mean() * 100.0) if not negative.empty else 0.0,
        "max_duration_days": int(max_streak),
    }


def _tail_risk_metrics(returns: pd.Series) -> dict[str, Any]:
    if returns.empty:
        return {
            "var_95": None,
            "cvar_95": None,
            "skew": None,
            "kurtosis": None,
            "best_day": None,
            "worst_day": None,
        }

    var_95 = returns.quantile(0.05)
    cvar_95 = returns[returns <= var_95].mean() if not returns[returns <= var_95].empty else np.nan

    return {
        "var_95": _safe_float(var_95 * 100.0),
        "cvar_95": _safe_float(cvar_95 * 100.0),
        "skew": _safe_float(returns.skew()),
        "kurtosis": _safe_float(returns.kurtosis()),
        "best_day": _safe_float(returns.max() * 100.0),
        "worst_day": _safe_float(returns.min() * 100.0),
    }


def _benchmark_metrics(returns: pd.Series, benchmark_returns: pd.Series) -> dict[str, Any]:
    if returns.empty or benchmark_returns.empty:
        return {
            "beta": None,
            "correlation": None,
            "tracking_error": None,
            "information_ratio": None,
        }

    aligned = pd.concat([returns.rename("strategy"), benchmark_returns.rename("benchmark")], axis=1).dropna()
    if aligned.empty:
        return {
            "beta": None,
            "correlation": None,
            "tracking_error": None,
            "information_ratio": None,
        }

    strategy = aligned["strategy"]
    bench = aligned["benchmark"]

    correlation = strategy.corr(bench)
    bench_var = bench.var()
    beta = strategy.cov(bench) / bench_var if bench_var > 0 else np.nan

    active = strategy - bench
    active_std = active.std()
    tracking_error = active_std * np.sqrt(252.0) if active_std > 0 else np.nan
    information_ratio = (active.mean() / active_std) * np.sqrt(252.0) if active_std > 0 else np.nan

    return {
        "beta": _safe_float(beta),
        "correlation": _safe_float(correlation),
        "tracking_error": _safe_float(tracking_error * 100.0),
        "information_ratio": _safe_float(information_ratio),
    }


def _turnover_metrics(holdings_history: Any) -> dict[str, Any]:
    if not isinstance(holdings_history, list) or len(holdings_history) == 0:
        return {
            "avg_positions": None,
            "avg_turnover": None,
            "rebalance_events": 0,
        }

    frame = pd.DataFrame(holdings_history)
    if frame.empty or "date" not in frame.columns or "ticker" not in frame.columns:
        return {
            "avg_positions": None,
            "avg_turnover": None,
            "rebalance_events": 0,
        }

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date", "ticker"])
    if frame.empty:
        return {
            "avg_positions": None,
            "avg_turnover": None,
            "rebalance_events": 0,
        }

    frame["ticker"] = frame["ticker"].map(_normalize_ticker)
    frame = frame[frame["ticker"] != ""]
    if frame.empty:
        return {
            "avg_positions": None,
            "avg_turnover": None,
            "rebalance_events": 0,
        }

    per_day: dict[pd.Timestamp, dict[str, float]] = {}
    has_weight_col = "weight" in frame.columns
    if has_weight_col:
        frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce")

    for dt, group in frame.groupby(frame["date"].dt.normalize()):
        day = pd.Timestamp(dt)
        tickers = sorted(set(group["ticker"].tolist()))
        if not tickers:
            per_day[day] = {}
            continue

        weights: dict[str, float]
        if has_weight_col:
            weighted = (
                group.dropna(subset=["weight"])
                .groupby("ticker")["weight"]
                .sum()
                .astype("float64")
            )
            weighted = weighted.replace([np.inf, -np.inf], np.nan).dropna()
            weighted = weighted[weighted > 0]
            total_weight = float(weighted.sum())
            if total_weight > 0:
                weights = {str(t): float(w / total_weight) for t, w in weighted.items()}
            else:
                equal = 1.0 / float(len(tickers))
                weights = {ticker: equal for ticker in tickers}
        else:
            equal = 1.0 / float(len(tickers))
            weights = {ticker: equal for ticker in tickers}

        per_day[day] = weights

    ordered_days = sorted(per_day.keys())
    if not ordered_days:
        return {
            "avg_positions": None,
            "avg_turnover": None,
            "rebalance_events": 0,
        }

    pos_counts = [len(per_day[day]) for day in ordered_days]
    turnovers: list[float] = []

    prev = per_day[ordered_days[0]]
    for day in ordered_days[1:]:
        current = per_day[day]
        if not current and not prev:
            prev = current
            continue

        universe = set(prev.keys()) | set(current.keys())
        # One-way turnover: 0.5 * sum(|w_t - w_t-1|), bounded in [0, 1] for long-only normalized weights.
        one_way_turnover = 0.5 * sum(
            abs(float(current.get(ticker, 0.0)) - float(prev.get(ticker, 0.0)))
            for ticker in universe
        )
        one_way_turnover = min(max(float(one_way_turnover), 0.0), 1.0)
        turnovers.append(one_way_turnover)
        prev = current

    return {
        "avg_positions": _safe_float(np.mean(pos_counts)) if pos_counts else None,
        "avg_turnover": _safe_float(np.mean(turnovers)) if turnovers else 0.0,
        "rebalance_events": int(len(turnovers)),
    }


def build_backtest_analytics_v2(
    returns: pd.Series | None,
    equity: pd.Series | None,
    benchmark_returns: pd.Series | None = None,
    holdings_history: Any = None,
) -> dict[str, Any]:
    strategy_returns = _to_series(returns)
    strategy_equity = _to_series(equity)
    benchmark = _to_series(benchmark_returns)

    if strategy_returns.empty or strategy_equity.empty:
        return {
            "summary": {
                "observations": 0,
                "start": None,
                "end": None,
                "positive_days": None,
                "negative_days": None,
                "flat_days": None,
            },
            "yearly": [],
            "monthly": [],
            "drawdown": _drawdown_metrics(pd.Series(dtype="float64")),
            "tail_risk": _tail_risk_metrics(pd.Series(dtype="float64")),
            "benchmark": _benchmark_metrics(pd.Series(dtype="float64"), pd.Series(dtype="float64")),
            "turnover": _turnover_metrics(holdings_history),
        }

    summary = {
        "observations": int(len(strategy_returns)),
        "start": strategy_returns.index.min().date().isoformat() if isinstance(strategy_returns.index, pd.DatetimeIndex) else None,
        "end": strategy_returns.index.max().date().isoformat() if isinstance(strategy_returns.index, pd.DatetimeIndex) else None,
        "positive_days": int((strategy_returns > 0).sum()),
        "negative_days": int((strategy_returns < 0).sum()),
        "flat_days": int((strategy_returns == 0).sum()),
    }

    return {
        "summary": summary,
        "yearly": _yearly_metrics(strategy_returns),
        "monthly": _monthly_returns(strategy_returns),
        "drawdown": _drawdown_metrics(strategy_equity),
        "tail_risk": _tail_risk_metrics(strategy_returns),
        "benchmark": _benchmark_metrics(strategy_returns, benchmark),
        "turnover": _turnover_metrics(holdings_history),
    }
