"use client";

import { useCallback, useEffect, useState } from "react";

interface RunHistoryItem {
    id: string;
    kind: string;
    status: string;
    created_at: string;
    updated_at: string;
    finished_at?: string;
}

interface RunHistoryListProps {
    kind: "factor_lab" | "signal_backtest" | "signal_construct" | "stock_filter";
    title?: string;
    limit?: number;
}

export default function RunHistoryList({ kind, title = "Recent Runs", limit = 8 }: RunHistoryListProps) {
    const [runs, setRuns] = useState<RunHistoryItem[]>([]);
    const [loading, setLoading] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);

    const loadRuns = useCallback(async () => {
        setLoading(true);
        setError(null);

        try {
            const response = await fetch(`/api/runs?kind=${encodeURIComponent(kind)}&limit=${limit}`, { cache: "no-store" });
            const payload = await response.json() as {
                runs?: RunHistoryItem[];
                error?: string;
            };

            if (!response.ok || payload.error) {
                throw new Error(payload.error || `Failed (${response.status})`);
            }

            setRuns(Array.isArray(payload.runs) ? payload.runs : []);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Failed to load run history.");
        } finally {
            setLoading(false);
        }
    }, [kind, limit]);

    useEffect(() => {
        void loadRuns();
        const timer = setInterval(() => {
            void loadRuns();
        }, 15_000);
        return () => clearInterval(timer);
    }, [loadRuns]);

    return (
        <div className="glass-card" style={{ padding: 12, marginBottom: 12 }}>
            <div style={{ display: "flex", justifyContent: "space-between", gap: 8, alignItems: "center", marginBottom: 8 }}>
                <h3 style={{ margin: 0, fontSize: "0.9rem" }}>{title}</h3>
                <button
                    onClick={() => void loadRuns()}
                    style={{
                        border: "1px solid var(--border-subtle)",
                        background: "var(--bg-secondary)",
                        color: "var(--text-primary)",
                        borderRadius: "var(--radius-sm)",
                        padding: "6px 10px",
                        fontSize: "0.76rem",
                        cursor: "pointer",
                    }}
                >
                    Refresh
                </button>
            </div>

            {error && (
                <div style={{ fontSize: "0.78rem", color: "var(--accent-rose)", marginBottom: 6 }}>{error}</div>
            )}

            {loading && runs.length === 0 && (
                <div style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>Loading run history...</div>
            )}

            {runs.length > 0 ? (
                <div style={{ overflowX: "auto" }}>
                    <table className="data-table">
                        <thead>
                            <tr>
                                <th>Run ID</th>
                                <th>Status</th>
                                <th>Updated</th>
                            </tr>
                        </thead>
                        <tbody>
                            {runs.map((run) => (
                                <tr key={run.id}>
                                    <td style={{ fontFamily: "var(--font-mono)", fontSize: "0.76rem" }}>{run.id}</td>
                                    <td>{run.status}</td>
                                    <td>{run.updated_at?.slice(0, 19).replace("T", " ")}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            ) : (!loading && !error && (
                <div style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>No runs yet for {kind}.</div>
            ))}
        </div>
    );
}
