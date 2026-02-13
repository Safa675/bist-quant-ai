import type { EngineEnvelope } from "@/lib/contracts/run";

export interface FactorParamSchema {
    key: string;
    label: string;
    type: "int" | "float" | "string" | "multi_select";
    default: number | string | string[];
    options?: Array<{ value: string; label: string }>;
    min?: number;
    max?: number;
}

export interface FactorCatalogEntry {
    name: string;
    label: string;
    description: string;
    rebalance_frequency: string;
    timeline: {
        start_date?: string;
        end_date?: string;
    };
    parameter_schema: FactorParamSchema[];
}

export interface FactorLabPayload {
    _mode?: "catalog" | "run";
    start_date?: string;
    end_date?: string;
    rebalance_frequency?: string;
    top_n?: number;
    factors?: Array<{
        name: string;
        enabled?: boolean;
        weight?: number;
        signal_params?: Record<string, unknown>;
    }>;
    portfolio_options?: Record<string, unknown>;
}

export interface FactorLabCatalogResult {
    factors: FactorCatalogEntry[];
    default_portfolio_options?: Record<string, unknown>;
}

export interface FactorLabBacktestResult {
    meta: {
        mode: string;
        as_of: string;
        start_date: string;
        end_date: string;
        rebalance_frequency: string;
        top_n: number;
        symbols_used: number;
        rows_used: number;
        execution_ms: number;
        factors: Array<{
            name: string;
            weight: number;
            signal_params: Record<string, unknown>;
        }>;
    };
    metrics: {
        cagr: number;
        sharpe: number;
        sortino: number;
        max_dd: number;
        total_return: number;
        win_rate: number;
        beta: number | null;
        rebalance_count: number;
        trade_count: number;
    };
    composite_top: Array<{ symbol: string; score: number | null }>;
    factor_top_symbols: Record<string, Array<{ symbol: string; score: number | null }>>;
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

export type FactorCatalogEnvelope = EngineEnvelope<FactorLabCatalogResult>;
export type FactorBacktestEnvelope = EngineEnvelope<FactorLabBacktestResult>;
