"use client";

import type { CSSProperties } from "react";

type RunStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled" | string;

interface RunStatusPanelProps {
    title?: string;
    runId?: string | null;
    status?: RunStatus | null;
    loading?: boolean;
    error?: string | null;
}

function statusStyle(status: RunStatus | null | undefined): CSSProperties {
    switch (status) {
        case "succeeded":
            return { color: "var(--accent-emerald)", background: "var(--accent-emerald-dim)" };
        case "failed":
        case "cancelled":
            return { color: "var(--accent-rose)", background: "rgba(244,63,94,0.14)" };
        case "running":
            return { color: "var(--accent-cyan)", background: "var(--accent-cyan-dim)" };
        default:
            return { color: "var(--text-secondary)", background: "var(--bg-secondary)" };
    }
}

export default function RunStatusPanel({
    title = "Run Status",
    runId,
    status,
    loading,
    error,
}: RunStatusPanelProps) {
    if (!runId && !error && !loading) {
        return null;
    }

    const resolvedStatus = status || (loading ? "queued" : "unknown");

    return (
        <div className="glass-card" style={{ padding: 12, marginBottom: 12, borderColor: "rgba(6,182,212,0.3)" }}>
            <div style={{ display: "flex", justifyContent: "space-between", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
                <div style={{ fontSize: "0.84rem", color: "var(--text-secondary)" }}>
                    <strong style={{ color: "var(--text-primary)" }}>{title}:</strong>{" "}
                    {runId ? <code style={{ color: "var(--text-primary)" }}>{runId}</code> : "pending"}
                </div>
                <span className="badge" style={statusStyle(resolvedStatus)}>
                    {String(resolvedStatus)}
                </span>
            </div>
            {error && (
                <div style={{ marginTop: 8, fontSize: "0.8rem", color: "var(--accent-rose)" }}>
                    {error}
                </div>
            )}
        </div>
    );
}
