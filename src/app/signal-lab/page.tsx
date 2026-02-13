"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import Navbar from "@/components/Navbar";
import { FlaskConical, SlidersHorizontal } from "lucide-react";

type TabKey = "models" | "technical";

interface UnifiedCatalogResponse {
    factor_catalog?: {
        factors?: Array<{ name: string }>;
    };
    error?: string;
}

export default function SignalLabPage() {
    const [activeTab, setActiveTab] = useState<TabKey>(() => {
        if (typeof window === "undefined") return "models";
        const tab = (new URLSearchParams(window.location.search).get("tab") || "models").toLowerCase();
        return tab === "technical" ? "technical" : "models";
    });
    const [signalCount, setSignalCount] = useState<number | null>(null);

    useEffect(() => {
        const loadCatalog = async () => {
            try {
                const response = await fetch("/api/catalog", { cache: "no-store" });
                const data = (await response.json()) as UnifiedCatalogResponse;
                if (!response.ok || data.error) {
                    return;
                }
                if (Array.isArray(data.factor_catalog?.factors)) {
                    setSignalCount(data.factor_catalog.factors.length);
                }
            } catch {
                // Ignore count failures and keep UX responsive.
            }
        };

        void loadCatalog();
    }, []);

    const frameSrc = useMemo(() => {
        if (activeTab === "technical") {
            return "/signal-construction?embed=1";
        }
        return "/factor-lab?embed=1";
    }, [activeTab]);

    return (
        <>
            <Navbar />
            <main className="page-compact" style={{ paddingTop: 88, paddingBottom: 16 }}>
                <div className="page-container">
                    <div style={{ marginBottom: 10, display: "flex", justifyContent: "space-between", gap: 10, flexWrap: "wrap" }}>
                        <div>
                            <h1 style={{ margin: 0, fontSize: "1.5rem", fontWeight: 800, letterSpacing: "-0.02em" }}>
                                Signal Lab
                            </h1>
                            <p style={{ margin: "4px 0 0", color: "var(--text-secondary)", fontSize: "0.85rem" }}>
                                Unified workspace for model signal combinations and technical signal construction.
                                {signalCount !== null ? ` Loaded model signals: ${signalCount}.` : ""}
                            </p>
                        </div>

                        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                            <Link
                                href="/stock-filter"
                                style={{
                                    border: "1px solid var(--border-subtle)",
                                    background: "var(--bg-secondary)",
                                    color: "var(--text-primary)",
                                    borderRadius: "var(--radius-sm)",
                                    padding: "8px 12px",
                                    fontSize: "0.8rem",
                                    textDecoration: "none",
                                }}
                            >
                                Stock Filter
                            </Link>
                        </div>
                    </div>

                    <div style={{ marginBottom: 10, display: "flex", gap: 8, flexWrap: "wrap" }}>
                        <button
                            onClick={() => setActiveTab("models")}
                            style={{
                                border: "1px solid var(--border-subtle)",
                                background: activeTab === "models" ? "var(--accent-emerald-dim)" : "var(--bg-secondary)",
                                color: activeTab === "models" ? "var(--accent-emerald)" : "var(--text-primary)",
                                borderRadius: "var(--radius-sm)",
                                padding: "8px 12px",
                                fontSize: "0.82rem",
                                cursor: "pointer",
                                display: "inline-flex",
                                gap: 6,
                                alignItems: "center",
                            }}
                        >
                            <FlaskConical size={14} />
                            Model Signal Mixer
                        </button>
                        <button
                            onClick={() => setActiveTab("technical")}
                            style={{
                                border: "1px solid var(--border-subtle)",
                                background: activeTab === "technical" ? "var(--accent-cyan-dim)" : "var(--bg-secondary)",
                                color: activeTab === "technical" ? "var(--accent-cyan)" : "var(--text-primary)",
                                borderRadius: "var(--radius-sm)",
                                padding: "8px 12px",
                                fontSize: "0.82rem",
                                cursor: "pointer",
                                display: "inline-flex",
                                gap: 6,
                                alignItems: "center",
                            }}
                        >
                            <SlidersHorizontal size={14} />
                            Technical Builder
                        </button>
                    </div>

                    <div className="glass-card" style={{ padding: 0, overflow: "hidden" }}>
                        <iframe
                            key={frameSrc}
                            src={frameSrc}
                            title="Signal Lab Workspace"
                            style={{
                                width: "100%",
                                border: "none",
                                minHeight: "1200px",
                                height: "calc(100vh - 190px)",
                                background: "var(--bg-primary)",
                            }}
                        />
                    </div>
                </div>
            </main>
        </>
    );
}
