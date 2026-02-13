import fallbackFactorCatalog from "@/lib/server/factorLabFallbackCatalog.json";
import type { FactorCatalogEntry, FactorLabCatalogResult } from "@/lib/contracts/factor";
import { executeFactorLabPython } from "@/lib/server/factorLabPython";

export interface SignalConstructionCatalog {
    universes: string[];
    periods: string[];
    intervals: string[];
    indicators: Array<{ key: string; label: string }>;
}

export interface UnifiedCatalog {
    generated_at: string;
    factor_catalog: {
        factors: FactorCatalogEntry[];
        default_portfolio_options: Record<string, unknown>;
        source: "engine" | "fallback";
        error?: string;
    };
    signal_construction: SignalConstructionCatalog;
}

const SIGNAL_CONSTRUCTION_CATALOG: SignalConstructionCatalog = {
    universes: ["XU030", "XU100", "XUTUM", "CUSTOM"],
    periods: ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
    intervals: ["1d"],
    indicators: [
        { key: "rsi", label: "RSI" },
        { key: "macd", label: "MACD Histogram" },
        { key: "bollinger", label: "Bollinger %B" },
        { key: "atr", label: "ATR (Cross-Sectional)" },
        { key: "stochastic", label: "Stochastic %K" },
        { key: "adx", label: "ADX (+DI/-DI trend)" },
        { key: "supertrend", label: "Supertrend Direction" },
    ],
};

const CATALOG_TTL_MS = 30_000;

let cache: { expires_at: number; value: UnifiedCatalog } | null = null;

function nowIso(): string {
    return new Date().toISOString();
}

function fallbackCatalog(error?: string): UnifiedCatalog {
    const raw = fallbackFactorCatalog as { factors?: FactorCatalogEntry[] };
    const factors = Array.isArray(raw.factors) ? raw.factors : [];

    return {
        generated_at: nowIso(),
        factor_catalog: {
            factors,
            default_portfolio_options: {},
            source: "fallback",
            error,
        },
        signal_construction: SIGNAL_CONSTRUCTION_CATALOG,
    };
}

export async function getUnifiedCatalog(options?: { forceRefresh?: boolean }): Promise<UnifiedCatalog> {
    const force = options?.forceRefresh === true;
    const now = Date.now();

    if (!force && cache && now < cache.expires_at) {
        return cache.value;
    }

    try {
        const envelope = await executeFactorLabPython<FactorLabCatalogResult>({ _mode: "catalog" });
        if (!envelope.error && envelope.result) {
            const value: UnifiedCatalog = {
                generated_at: nowIso(),
                factor_catalog: {
                    factors: Array.isArray(envelope.result.factors) ? envelope.result.factors : [],
                    default_portfolio_options: envelope.result.default_portfolio_options || {},
                    source: "engine",
                },
                signal_construction: SIGNAL_CONSTRUCTION_CATALOG,
            };
            cache = {
                expires_at: now + CATALOG_TTL_MS,
                value,
            };
            return value;
        }

        const fallback = fallbackCatalog(envelope.error?.message || "Factor engine catalog failed.");
        cache = {
            expires_at: now + CATALOG_TTL_MS,
            value: fallback,
        };
        return fallback;
    } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        const fallback = fallbackCatalog(message);
        cache = {
            expires_at: now + CATALOG_TTL_MS,
            value: fallback,
        };
        return fallback;
    }
}

export async function getFactorCatalog(options?: { forceRefresh?: boolean }): Promise<{
    factors: FactorCatalogEntry[];
    default_portfolio_options: Record<string, unknown>;
    source: "engine" | "fallback";
    error?: string;
}> {
    const unified = await getUnifiedCatalog(options);
    return unified.factor_catalog;
}

export function getSignalConstructionCatalog(): SignalConstructionCatalog {
    return SIGNAL_CONSTRUCTION_CATALOG;
}
