import type { EngineEnvelope } from "@/lib/contracts/run";

export interface StockFilterPayload {
    _mode?: "meta" | "run";
    template?: string;
    sector?: string;
    index?: string;
    recommendation?: string;
    sort_by?: string;
    sort_desc?: boolean;
    limit?: number;
    columns?: string[];
    filters?: Record<string, { min?: number | null; max?: number | null }>;
    percentile_filters?: Record<string, { min_pct?: number | null; max_pct?: number | null }>;
}

export interface StockFilterMetaResult {
    templates: string[];
    filters: Array<{ key: string; label: string; group: string }>;
    indexes: string[];
    recommendations: string[];
    default_sort_by: string;
    default_sort_desc: boolean;
    filter_mode?: string;
}

export interface StockFilterRunResult {
    meta: {
        as_of: string;
        execution_ms: number;
        total_matches: number;
        returned_rows: number;
        template?: string | null;
        sector?: string | null;
        index?: string | null;
        recommendation?: string | null;
        sort_by: string;
        sort_desc: boolean;
    };
    columns: Array<{ key: string; label: string }>;
    rows: Array<Record<string, string | number | boolean | null>>;
    applied_filters: Array<{
        key: string;
        label: string;
        min: number | null;
        max: number | null;
    }>;
    applied_percentile_filters?: Array<{
        key: string;
        label: string;
        min_pct: number | null;
        max_pct: number | null;
    }>;
}

export type StockFilterMetaEnvelope = EngineEnvelope<StockFilterMetaResult>;
export type StockFilterRunEnvelope = EngineEnvelope<StockFilterRunResult>;
