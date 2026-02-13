"use client";

import { useEffect, useMemo, useState, type CSSProperties } from "react";
import Link from "next/link";
import Navbar from "@/components/Navbar";
import { Filter, PlayCircle, RotateCcw } from "lucide-react";

interface FilterFieldDef {
    key: string;
    label: string;
    group: string;
}

interface StockFilterMeta {
    templates: string[];
    filters: FilterFieldDef[];
    indexes: string[];
    recommendations: string[];
    default_sort_by: string;
    default_sort_desc: boolean;
    error?: string;
}

interface FilterBound {
    min: string;
    max: string;
}

type FilterBoundState = Record<string, FilterBound>;

interface StockFilterColumn {
    key: string;
    label: string;
}

interface StockFilterResult {
    meta: {
        as_of: string;
        execution_ms: number;
        total_matches: number;
        returned_rows: number;
        sort_by: string;
        sort_desc: boolean;
    };
    columns: StockFilterColumn[];
    rows: Array<Record<string, string | number | null>>;
    applied_filters: Array<{
        key: string;
        label: string;
        min: number | null;
        max: number | null;
    }>;
    error?: string;
}

function parseOptionalNumber(raw: string): number | null {
    const text = raw.trim();
    if (!text) return null;
    const parsed = Number(text);
    if (!Number.isFinite(parsed)) return null;
    return parsed;
}

function formatValue(value: string | number | null): string {
    if (value === null || value === undefined) return "—";
    if (typeof value === "number") {
        if (!Number.isFinite(value)) return "—";
        const abs = Math.abs(value);
        if (abs >= 1_000_000) return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
        if (abs >= 1000) return value.toLocaleString(undefined, { maximumFractionDigits: 1 });
        if (abs >= 1) return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
        return value.toLocaleString(undefined, { maximumFractionDigits: 4 });
    }
    return String(value);
}

export default function StockFilterPage() {
    const [meta, setMeta] = useState<StockFilterMeta | null>(null);
    const [metaError, setMetaError] = useState<string | null>(null);

    const [template, setTemplate] = useState<string>("");
    const [sector, setSector] = useState<string>("");
    const [indexName, setIndexName] = useState<string>("XU100");
    const [recommendation, setRecommendation] = useState<string>("");
    const [sortBy, setSortBy] = useState<string>("upside_potential");
    const [sortDesc, setSortDesc] = useState<boolean>(true);
    const [limit, setLimit] = useState<number>(100);

    const [bounds, setBounds] = useState<FilterBoundState>({});

    const [running, setRunning] = useState<boolean>(false);
    const [runError, setRunError] = useState<string | null>(null);
    const [result, setResult] = useState<StockFilterResult | null>(null);

    useEffect(() => {
        const loadMeta = async () => {
            setMetaError(null);
            try {
                const response = await fetch("/api/stock-filter", { cache: "no-store" });
                const data = (await response.json()) as StockFilterMeta;
                if (!response.ok || data.error) {
                    throw new Error(data.error || `Failed (${response.status})`);
                }

                setMeta(data);
                setSortBy(data.default_sort_by || "upside_potential");
                setSortDesc(Boolean(data.default_sort_desc));

                const initialBounds: FilterBoundState = {};
                for (const field of data.filters || []) {
                    initialBounds[field.key] = { min: "", max: "" };
                }
                setBounds(initialBounds);
            } catch (err) {
                setMetaError(err instanceof Error ? err.message : "Failed to load stock filter metadata.");
            }
        };

        void loadMeta();
    }, []);

    const groupedFields = useMemo(() => {
        const groups = new Map<string, FilterFieldDef[]>();
        for (const field of meta?.filters || []) {
            const current = groups.get(field.group) || [];
            current.push(field);
            groups.set(field.group, current);
        }
        return Array.from(groups.entries());
    }, [meta]);

    const updateBound = (key: string, side: "min" | "max", value: string) => {
        setBounds((prev) => ({
            ...prev,
            [key]: {
                ...(prev[key] || { min: "", max: "" }),
                [side]: value,
            },
        }));
    };

    const clearBounds = () => {
        setBounds((prev) => {
            const next: FilterBoundState = {};
            for (const key of Object.keys(prev)) {
                next[key] = { min: "", max: "" };
            }
            return next;
        });
        setRunError(null);
    };

    const runFilter = async () => {
        setRunning(true);
        setRunError(null);

        try {
            const filters = Object.entries(bounds).reduce((acc, [key, value]) => {
                const min = parseOptionalNumber(value.min);
                const max = parseOptionalNumber(value.max);
                if (min === null && max === null) {
                    return acc;
                }
                acc[key] = { min, max };
                return acc;
            }, {} as Record<string, { min: number | null; max: number | null }>);

            const payload = {
                template,
                sector,
                index: indexName,
                recommendation,
                sort_by: sortBy,
                sort_desc: sortDesc,
                limit,
                filters,
            };

            const response = await fetch("/api/stock-filter", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            const data: StockFilterResult = await response.json();

            if (!response.ok || data.error) {
                throw new Error(data.error || `Request failed (${response.status})`);
            }

            setResult(data);
        } catch (err) {
            setRunError(err instanceof Error ? err.message : "Stock filter failed.");
        } finally {
            setRunning(false);
        }
    };

    return (
        <>
            <Navbar />
            <main className="page-compact" style={{ paddingTop: 88 }}>
                <div className="page-container">
                    <div style={{ marginBottom: 14, display: "flex", justifyContent: "space-between", gap: 10, flexWrap: "wrap" }}>
                        <div>
                            <h1 style={{ margin: 0, fontSize: "1.5rem", fontWeight: 800, letterSpacing: "-0.02em" }}>
                                Stock Filter
                            </h1>
                            <p style={{ margin: "4px 0 0", color: "var(--text-secondary)", fontSize: "0.85rem" }}>
                                Manual BIST screener with fundamentals and valuation filters powered by <code>borsapy</code>.
                            </p>
                        </div>
                        <Link href="/signal-lab" style={secondaryBtn}>
                            Open Signal Lab
                        </Link>
                    </div>

                    {metaError && (
                        <div className="glass-card" style={{ padding: 14, marginBottom: 14, borderColor: "rgba(244,63,94,0.4)" }}>
                            <strong style={{ color: "var(--accent-rose)" }}>Metadata error:</strong> {metaError}
                        </div>
                    )}

                    <div className="glass-card" style={{ padding: 12, marginBottom: 10 }}>
                        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10, marginBottom: 8, flexWrap: "wrap" }}>
                            <h2 style={{ margin: 0, fontSize: "0.95rem", display: "inline-flex", gap: 6, alignItems: "center" }}>
                                <Filter size={14} /> Screener Settings
                            </h2>
                            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                                <button onClick={clearBounds} style={secondaryBtn}>
                                    <RotateCcw size={14} /> Clear Filters
                                </button>
                                <button className="btn-primary" onClick={runFilter} disabled={running || !meta}>
                                    <PlayCircle size={16} />
                                    {running ? "Filtering..." : "Run Filter"}
                                </button>
                            </div>
                        </div>

                        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))", gap: 8, marginBottom: 8 }}>
                            <label style={{ display: "grid", gap: 6 }}>
                                <span className="metric-label">Template</span>
                                <select value={template} onChange={(e) => setTemplate(e.target.value)} style={inputStyle}>
                                    <option value="">none</option>
                                    {(meta?.templates || []).map((item) => (
                                        <option key={item} value={item}>{item}</option>
                                    ))}
                                </select>
                            </label>

                            <label style={{ display: "grid", gap: 6 }}>
                                <span className="metric-label">Index</span>
                                <select value={indexName} onChange={(e) => setIndexName(e.target.value)} style={inputStyle}>
                                    {(meta?.indexes || ["XU100"]).map((item) => (
                                        <option key={item} value={item}>{item}</option>
                                    ))}
                                </select>
                            </label>

                            <label style={{ display: "grid", gap: 6 }}>
                                <span className="metric-label">Sector (optional)</span>
                                <input value={sector} onChange={(e) => setSector(e.target.value)} placeholder="Bankacılık" style={inputStyle} />
                            </label>

                            <label style={{ display: "grid", gap: 6 }}>
                                <span className="metric-label">Recommendation</span>
                                <select value={recommendation} onChange={(e) => setRecommendation(e.target.value)} style={inputStyle}>
                                    <option value="">any</option>
                                    {(meta?.recommendations || []).map((item) => (
                                        <option key={item} value={item}>{item}</option>
                                    ))}
                                </select>
                            </label>

                            <label style={{ display: "grid", gap: 6 }}>
                                <span className="metric-label">Sort By</span>
                                <select value={sortBy} onChange={(e) => setSortBy(e.target.value)} style={inputStyle}>
                                    {(meta?.filters || []).map((field) => (
                                        <option key={field.key} value={field.key}>{field.label}</option>
                                    ))}
                                </select>
                            </label>

                            <label style={{ display: "grid", gap: 6 }}>
                                <span className="metric-label">Limit</span>
                                <input
                                    type="number"
                                    min={1}
                                    max={500}
                                    value={limit}
                                    onChange={(e) => setLimit(Number(e.target.value) || 100)}
                                    style={inputStyle}
                                />
                            </label>
                        </div>

                        <label style={{ display: "inline-flex", alignItems: "center", gap: 8, marginBottom: 2 }}>
                            <input type="checkbox" checked={sortDesc} onChange={(e) => setSortDesc(e.target.checked)} />
                            <span className="metric-label">Sort descending</span>
                        </label>
                    </div>

                    {(groupedFields.length > 0) && (
                        <div className="glass-card" style={{ padding: 12, marginBottom: 10 }}>
                            <h2 style={{ margin: "0 0 8px", fontSize: "0.95rem" }}>Fundamental Filters</h2>

                            <div style={{ display: "grid", gap: 10 }}>
                                {groupedFields.map(([group, fields]) => (
                                    <div key={group} style={{ border: "1px solid var(--border-subtle)", borderRadius: "var(--radius-md)", padding: 10 }}>
                                        <div style={{ marginBottom: 8, fontSize: "0.78rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em" }}>
                                            {group}
                                        </div>
                                        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 8 }}>
                                            {fields.map((field) => {
                                                const bound = bounds[field.key] || { min: "", max: "" };
                                                return (
                                                    <div key={field.key} style={{ border: "1px solid var(--border-subtle)", borderRadius: "var(--radius-sm)", padding: 8 }}>
                                                        <div style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginBottom: 6 }}>{field.label}</div>
                                                        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
                                                            <input
                                                                type="number"
                                                                value={bound.min}
                                                                onChange={(e) => updateBound(field.key, "min", e.target.value)}
                                                                placeholder="min"
                                                                style={inputStyle}
                                                            />
                                                            <input
                                                                type="number"
                                                                value={bound.max}
                                                                onChange={(e) => updateBound(field.key, "max", e.target.value)}
                                                                placeholder="max"
                                                                style={inputStyle}
                                                            />
                                                        </div>
                                                    </div>
                                                );
                                            })}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {runError && (
                        <div className="glass-card" style={{ padding: 14, marginBottom: 14, borderColor: "rgba(244,63,94,0.4)" }}>
                            <strong style={{ color: "var(--accent-rose)" }}>Filter failed:</strong> {runError}
                        </div>
                    )}

                    {result && (
                        <div className="glass-card" style={{ padding: 12 }}>
                            <div style={{ display: "flex", justifyContent: "space-between", gap: 10, flexWrap: "wrap", marginBottom: 10 }}>
                                <div>
                                    <h2 style={{ margin: 0, fontSize: "1.0rem" }}>Filter Results</h2>
                                    <p style={{ margin: "4px 0 0", color: "var(--text-muted)", fontSize: "0.8rem" }}>
                                        {result.meta.returned_rows} returned / {result.meta.total_matches} matched · {result.meta.execution_ms}ms
                                    </p>
                                </div>
                                <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                                    <span className="badge" style={{ background: "var(--bg-secondary)" }}>
                                        Sort: {result.meta.sort_by} {result.meta.sort_desc ? "desc" : "asc"}
                                    </span>
                                    <span className="badge" style={{ background: "var(--bg-secondary)" }}>
                                        Applied filters: {result.applied_filters.length}
                                    </span>
                                </div>
                            </div>

                            {result.applied_filters.length > 0 && (
                                <div style={{ marginBottom: 10, display: "flex", gap: 6, flexWrap: "wrap" }}>
                                    {result.applied_filters.map((filter) => (
                                        <span key={filter.key} className="badge" style={{ background: "var(--accent-cyan-dim)", color: "var(--accent-cyan)" }}>
                                            {filter.label}: {filter.min ?? "-∞"} → {filter.max ?? "+∞"}
                                        </span>
                                    ))}
                                </div>
                            )}

                            <div style={{ overflowX: "auto" }}>
                                <table className="data-table">
                                    <thead>
                                        <tr>
                                            <th>#</th>
                                            {result.columns.map((column) => (
                                                <th key={column.key}>{column.label}</th>
                                            ))}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {result.rows.map((row, idx) => (
                                            <tr key={`${String(row.symbol || "row")}-${idx}`}>
                                                <td>{idx + 1}</td>
                                                {result.columns.map((column) => (
                                                    <td key={`${idx}-${column.key}`} style={column.key === "symbol" ? { fontWeight: 700 } : undefined}>
                                                        {formatValue(row[column.key] ?? null)}
                                                    </td>
                                                ))}
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    )}
                </div>
            </main>
        </>
    );
}

const inputStyle: CSSProperties = {
    width: "100%",
    border: "1px solid var(--border-subtle)",
    background: "rgba(0,0,0,0.3)",
    color: "var(--text-primary)",
    borderRadius: "var(--radius-sm)",
    padding: "8px 10px",
    fontSize: "0.85rem",
    outline: "none",
};

const secondaryBtn: CSSProperties = {
    border: "1px solid var(--border-subtle)",
    background: "var(--bg-secondary)",
    color: "var(--text-primary)",
    borderRadius: "var(--radius-sm)",
    padding: "8px 12px",
    fontSize: "0.8rem",
    fontWeight: 600,
    cursor: "pointer",
    display: "inline-flex",
    gap: 6,
    alignItems: "center",
    textDecoration: "none",
};
