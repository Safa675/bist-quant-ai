import { execFile } from "child_process";
import { existsSync, readFileSync } from "fs";
import { join } from "path";
import { promisify } from "util";
import { readArtifactByPath } from "@/lib/server/artifactStore";
import { listRuns } from "@/lib/server/runStore";
import { listPublishedSignals, type StoredPublishedSignal } from "@/lib/server/signalStore";

const execFileAsync = promisify(execFile);

const PROJECT_ROOT = process.cwd();
const GENERATOR_PATH = join(PROJECT_ROOT, "dashboard", "generate_dashboard_data.py");
const SNAPSHOT_PATH = join(PROJECT_ROOT, "public", "data", "dashboard_data.json");
const PYTHON_CANDIDATES = ["python3", "python"] as const;
const REFRESH_THROTTLE_MS = 10_000;

let lastRefreshAtMs = 0;

const LAB_RUN_KINDS = new Set(["factor_lab", "signal_backtest"]);
const LAB_RUNS_PER_KIND = 3;

function isRecord(value: unknown): value is Record<string, unknown> {
    return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function toFiniteNumber(value: unknown): number | null {
    if (typeof value === "number" && Number.isFinite(value)) return value;
    if (typeof value === "string") {
        const parsed = Number(value);
        if (Number.isFinite(parsed)) return parsed;
    }
    return null;
}

async function collectRecentLabRuns(): Promise<{
    runs: Array<Record<string, unknown>>;
    latestByKind: Record<string, Record<string, unknown>>;
}> {
    const { runs } = await listRuns({ status: "succeeded", limit: 300 });
    const filtered = runs.filter((run) => LAB_RUN_KINDS.has(run.kind));
    const targetKinds = Array.from(LAB_RUN_KINDS);

    const countsByKind: Record<string, number> = {};
    const summaries: Array<Record<string, unknown>> = [];
    const latestByKind: Record<string, Record<string, unknown>> = {};

    for (const run of filtered) {
        if (targetKinds.every((kind) => (countsByKind[kind] || 0) >= LAB_RUNS_PER_KIND)) {
            break;
        }

        const currentCount = countsByKind[run.kind] || 0;
        if (currentCount >= LAB_RUNS_PER_KIND) {
            continue;
        }

        const artifactPath =
            run.meta && typeof run.meta.artifact_path === "string"
                ? run.meta.artifact_path
                : "";
        if (!artifactPath) {
            continue;
        }

        const artifact = await readArtifactByPath(artifactPath).catch(() => null);
        if (!isRecord(artifact)) {
            continue;
        }

        const metrics = isRecord(artifact.metrics) ? artifact.metrics : {};
        const artifactMeta = isRecord(artifact.meta) ? artifact.meta : {};
        const holdings = Array.isArray(artifact.current_holdings)
            ? artifact.current_holdings.map((item) => String(item))
            : [];

        const summary: Record<string, unknown> = {
            run_id: run.id,
            kind: run.kind,
            status: run.status,
            created_at: run.created_at,
            finished_at: run.finished_at || run.updated_at,
            as_of: typeof artifactMeta.as_of === "string" ? artifactMeta.as_of : null,
            cagr: toFiniteNumber(metrics.cagr),
            sharpe: toFiniteNumber(metrics.sharpe),
            max_dd: toFiniteNumber(metrics.max_dd),
            total_return: toFiniteNumber(metrics.total_return),
            holdings_count: holdings.length,
            holdings_preview: holdings.slice(0, 12),
            analytics_summary: isRecord(artifact.analytics_v2) && isRecord(artifact.analytics_v2.summary)
                ? artifact.analytics_v2.summary
                : null,
        };

        summaries.push(summary);
        countsByKind[run.kind] = currentCount + 1;
        if (!latestByKind[run.kind]) {
            latestByKind[run.kind] = summary;
        }
    }

    return {
        runs: summaries,
        latestByKind,
    };
}

function mergePublishedSignals(
    snapshot: Record<string, unknown>,
    publishedSignals: StoredPublishedSignal[]
): Record<string, unknown> {
    if (!publishedSignals.length) {
        return snapshot;
    }

    const signals = Array.isArray(snapshot.signals)
        ? [...(snapshot.signals as Array<Record<string, unknown>>)]
        : [];
    const holdingsRoot =
        snapshot.holdings && typeof snapshot.holdings === "object" && !Array.isArray(snapshot.holdings)
            ? (snapshot.holdings as Record<string, string[]>)
            : {};
    const holdings: Record<string, string[]> = { ...holdingsRoot };
    const allPublishedNames = new Set<string>(
        publishedSignals
            .map((item) => item.signal?.name)
            .filter((name): name is string => typeof name === "string" && name.length > 0)
    );

    const filteredSignals = signals.filter((item) => !allPublishedNames.has(String(item?.name || "")));
    for (const name of allPublishedNames) {
        delete holdings[name];
    }

    const enabledPublishedSignals = publishedSignals.filter(
        (item) => item.enabled !== false && item.signal?.enabled !== false
    );
    for (const published of enabledPublishedSignals) {
        const signalName = published.signal.name;
        filteredSignals.push({ ...published.signal, enabled: true });
        holdings[signalName] = [...published.holdings];
    }

    filteredSignals.sort((a, b) => {
        const aCagr = typeof a.cagr === "number" ? a.cagr : Number.NEGATIVE_INFINITY;
        const bCagr = typeof b.cagr === "number" ? b.cagr : Number.NEGATIVE_INFINITY;
        return bCagr - aCagr;
    });

    return {
        ...snapshot,
        signals: filteredSignals,
        holdings,
        active_signals: filteredSignals.filter((item) => item?.enabled !== false).length,
        displayed_signals: filteredSignals.length,
        published_signals_count: publishedSignals.length,
        published_signals_enabled_count: enabledPublishedSignals.length,
    };
}

function readSnapshot(): Record<string, unknown> {
    if (!existsSync(SNAPSHOT_PATH)) {
        throw new Error(`Dashboard snapshot not found: ${SNAPSHOT_PATH}`);
    }

    const raw = readFileSync(SNAPSHOT_PATH, "utf-8");
    const parsed: unknown = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
        throw new Error("Dashboard snapshot root must be a JSON object.");
    }
    return parsed as Record<string, unknown>;
}

async function runGenerator(): Promise<void> {
    if (!existsSync(GENERATOR_PATH)) {
        throw new Error(`Dashboard generator not found: ${GENERATOR_PATH}`);
    }

    let lastError: unknown = null;
    for (const pythonCmd of PYTHON_CANDIDATES) {
        try {
            await execFileAsync(pythonCmd, [GENERATOR_PATH], {
                cwd: PROJECT_ROOT,
                timeout: 120_000,
                maxBuffer: 10 * 1024 * 1024,
            });
            return;
        } catch (error) {
            lastError = error;
        }
    }

    const detail = lastError instanceof Error ? lastError.message : String(lastError);
    throw new Error(`Failed to run dashboard generator with python3/python: ${detail}`);
}

export async function loadDashboardData(options?: {
    refresh?: boolean;
    force?: boolean;
}): Promise<{
    data: Record<string, unknown>;
    refreshed: boolean;
    refreshError?: string;
}> {
    const wantsRefresh = options?.refresh === true;
    const forceRefresh = options?.force === true;
    let refreshed = false;
    let refreshError: string | undefined;

    if (wantsRefresh) {
        const now = Date.now();
        const canRefresh = forceRefresh || now - lastRefreshAtMs >= REFRESH_THROTTLE_MS;
        if (canRefresh) {
            try {
                await runGenerator();
                refreshed = true;
                lastRefreshAtMs = now;
            } catch (error) {
                refreshError = error instanceof Error ? error.message : String(error);
            }
        }
    }

    const snapshot = readSnapshot();
    const publishedSignals = await listPublishedSignals().catch(() => []);
    const baseData = mergePublishedSignals(snapshot, publishedSignals);
    const recentLabRuns = await collectRecentLabRuns().catch(() => ({
        runs: [] as Array<Record<string, unknown>>,
        latestByKind: {} as Record<string, Record<string, unknown>>,
    }));
    const data = {
        ...baseData,
        recent_lab_runs: recentLabRuns.runs,
        recent_lab_runs_count: recentLabRuns.runs.length,
        latest_lab_run_by_kind: recentLabRuns.latestByKind,
    };

    return {
        data,
        refreshed,
        refreshError,
    };
}
