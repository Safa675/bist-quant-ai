import type { EngineEnvelope } from "@/lib/contracts/run";

export interface IndicatorPayload {
    enabled?: boolean;
    params?: Record<string, number>;
}

export interface SignalConstructionPayload {
    universe?: string;
    symbols?: string[] | string;
    period?: string;
    interval?: string;
    max_symbols?: number;
    top_n?: number;
    buy_threshold?: number;
    sell_threshold?: number;
    indicators?: Record<string, IndicatorPayload>;
    _mode?: "construct" | "backtest";
}

export interface SignalConstructResult {
    meta: {
        mode: string;
        universe: string;
        period: string;
        interval: string;
        symbols_used: number;
        rows_used: number;
        as_of: string;
        indicators: string[];
        execution_ms: number;
    };
    indicator_summaries: Array<{
        name: string;
        buy_count: number;
        sell_count: number;
        hold_count: number;
    }>;
    signals: Array<{
        symbol: string;
        action: string;
        combined_score: number | null;
        buy_votes: number | null;
        sell_votes: number | null;
        hold_votes: number | null;
        indicator_values: Record<string, number | null>;
        indicator_signals: Record<string, number | null>;
    }>;
}

export interface SignalBacktestResult {
    meta: {
        mode: string;
        universe: string;
        period: string;
        interval: string;
        symbols_used: number;
        rows_used: number;
        as_of: string;
        indicators: string[];
        buy_threshold: number;
        sell_threshold: number;
        max_positions: number;
        execution_ms: number;
    };
    metrics: {
        cagr: number;
        sharpe: number;
        max_dd: number;
        ytd: number;
        total_return: number;
        volatility: number;
        win_rate: number;
        beta: number | null;
        last_rebalance: string;
    };
    signals: SignalConstructResult["signals"];
    indicator_summaries: SignalConstructResult["indicator_summaries"];
    current_holdings: string[];
    equity_curve: Array<{ date: string; value: number }>;
    benchmark_curve: Array<{ date: string; value: number }>;
    analytics_v2?: {
        summary?: Record<string, unknown>;
        yearly?: Array<Record<string, unknown>>;
        monthly?: Array<Record<string, unknown>>;
        drawdown?: Record<string, unknown>;
        tail_risk?: Record<string, unknown>;
        benchmark?: Record<string, unknown>;
        turnover?: Record<string, unknown>;
    };
}

export type SignalConstructEnvelope = EngineEnvelope<SignalConstructResult>;
export type SignalBacktestEnvelope = EngineEnvelope<SignalBacktestResult>;
