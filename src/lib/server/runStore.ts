import { existsSync } from "fs";
import { mkdir, readFile, rename, writeFile } from "fs/promises";
import { saveArtifact, type StoredArtifact } from "@/lib/server/artifactStore";
import {
    type EngineError,
    type RunKind,
    type RunRecord,
    type RunStatus,
    isObjectRecord,
} from "@/lib/contracts/run";
import { DATA_DIR, RUN_STORE_PATH } from "@/lib/server/storagePaths";

interface RunStoreState {
    runs: RunRecord[];
}

const DEFAULT_STATE: RunStoreState = {
    runs: [],
};
let mutationChain: Promise<void> = Promise.resolve();

function nowIso(): string {
    return new Date().toISOString();
}

function runId(kind: RunKind): string {
    const prefix = String(kind || "run").replace(/[^a-z0-9]+/gi, "_").toLowerCase();
    const rand = Math.random().toString(36).slice(2, 10);
    return `${prefix}_${Date.now()}_${rand}`;
}

function parseState(raw: unknown): RunStoreState {
    if (!isObjectRecord(raw)) {
        return { ...DEFAULT_STATE };
    }

    const rawRuns = Array.isArray(raw.runs) ? raw.runs : [];
    const runs: RunRecord[] = [];

    for (const item of rawRuns) {
        if (!isObjectRecord(item)) continue;
        if (typeof item.id !== "string" || !item.id) continue;
        if (typeof item.kind !== "string" || !item.kind) continue;
        if (typeof item.status !== "string" || !item.status) continue;
        if (typeof item.created_at !== "string" || !item.created_at) continue;
        if (typeof item.updated_at !== "string" || !item.updated_at) continue;

        runs.push({
            id: item.id,
            kind: item.kind as RunKind,
            status: item.status as RunStatus,
            created_at: item.created_at,
            updated_at: item.updated_at,
            started_at: typeof item.started_at === "string" ? item.started_at : undefined,
            finished_at: typeof item.finished_at === "string" ? item.finished_at : undefined,
            request: isObjectRecord(item.request) ? item.request : {},
            meta: isObjectRecord(item.meta) ? item.meta : undefined,
            artifact_id: typeof item.artifact_id === "string" ? item.artifact_id : undefined,
            error: isObjectRecord(item.error)
                ? {
                    code: typeof item.error.code === "string" ? item.error.code : "RUN_ERROR",
                    message: typeof item.error.message === "string" ? item.error.message : "Unknown run error",
                    details: item.error.details,
                }
                : undefined,
        });
    }

    return { runs };
}

async function ensureStoreDir(): Promise<void> {
    await mkdir(DATA_DIR, { recursive: true });
}

async function readState(): Promise<RunStoreState> {
    await ensureStoreDir();
    if (!existsSync(RUN_STORE_PATH)) {
        return { ...DEFAULT_STATE };
    }

    try {
        const raw = await readFile(RUN_STORE_PATH, "utf-8");
        return parseState(JSON.parse(raw));
    } catch {
        return { ...DEFAULT_STATE };
    }
}

async function writeState(state: RunStoreState): Promise<void> {
    await ensureStoreDir();
    const tmpPath = `${RUN_STORE_PATH}.${process.pid}.${Date.now()}.tmp`;
    await writeFile(tmpPath, JSON.stringify(state, null, 2), "utf-8");
    await rename(tmpPath, RUN_STORE_PATH);
}

async function withExclusiveMutation<T>(mutate: (state: RunStoreState) => Promise<T> | T): Promise<T> {
    const previous = mutationChain;
    let release!: () => void;
    mutationChain = new Promise<void>((resolve) => {
        release = resolve;
    });

    await previous;
    try {
        const state = await readState();
        const result = await mutate(state);
        await writeState(state);
        return result;
    } finally {
        release();
    }
}

function sortRuns(runs: RunRecord[]): RunRecord[] {
    return [...runs].sort((a, b) => b.updated_at.localeCompare(a.updated_at));
}

function mergeMeta(current: Record<string, unknown> | undefined, patch: Record<string, unknown> | undefined) {
    if (!patch) return current;
    return {
        ...(current || {}),
        ...patch,
    };
}

export async function createRun(input: {
    kind: RunKind;
    request: Record<string, unknown>;
    status?: RunStatus;
    meta?: Record<string, unknown>;
    id?: string;
}): Promise<RunRecord> {
    return withExclusiveMutation((state) => {
        const now = nowIso();
        const created: RunRecord = {
            id: input.id?.trim() || runId(input.kind),
            kind: input.kind,
            status: input.status || "queued",
            created_at: now,
            updated_at: now,
            request: input.request,
            meta: input.meta,
        };

        if (created.status === "running") {
            created.started_at = now;
        }

        state.runs.push(created);
        return created;
    });
}

export async function listRuns(options?: {
    kind?: RunKind;
    status?: RunStatus;
    limit?: number;
    offset?: number;
}): Promise<{ runs: RunRecord[]; total: number }> {
    const state = await readState();
    let runs = sortRuns(state.runs);

    if (options?.kind) {
        runs = runs.filter((run) => run.kind === options.kind);
    }
    if (options?.status) {
        runs = runs.filter((run) => run.status === options.status);
    }

    const total = runs.length;
    const offset = options?.offset && options.offset > 0 ? options.offset : 0;
    const limit = options?.limit && options.limit > 0 ? options.limit : 100;

    return {
        runs: runs.slice(offset, offset + limit),
        total,
    };
}

export async function getRun(id: string): Promise<RunRecord | null> {
    const state = await readState();
    return state.runs.find((run) => run.id === id) || null;
}

export async function updateRun(input: {
    id: string;
    status?: RunStatus;
    meta?: Record<string, unknown>;
    artifact_id?: string;
    error?: EngineError | null;
}): Promise<RunRecord | null> {
    return withExclusiveMutation((state) => {
        const idx = state.runs.findIndex((run) => run.id === input.id);
        if (idx < 0) {
            return null;
        }

        const current = state.runs[idx];
        const next: RunRecord = {
            ...current,
            status: input.status || current.status,
            updated_at: nowIso(),
            meta: mergeMeta(current.meta, input.meta),
            artifact_id: input.artifact_id ?? current.artifact_id,
            error: input.error === null ? undefined : (input.error ?? current.error),
        };

        if (next.status === "running" && !next.started_at) {
            next.started_at = nowIso();
        }
        if (next.status === "queued") {
            next.started_at = undefined;
            next.finished_at = undefined;
        }

        if (["succeeded", "failed", "cancelled"].includes(next.status) && !next.finished_at) {
            next.finished_at = nowIso();
        }

        state.runs[idx] = next;
        return next;
    });
}

export async function markRunRunning(id: string, meta?: Record<string, unknown>): Promise<RunRecord | null> {
    return updateRun({ id, status: "running", meta });
}

export async function markRunFailed(
    id: string,
    error: EngineError,
    meta?: Record<string, unknown>
): Promise<RunRecord | null> {
    return updateRun({
        id,
        status: "failed",
        error,
        meta,
    });
}

export async function markRunCancelled(id: string, reason?: string): Promise<RunRecord | null> {
    return updateRun({
        id,
        status: "cancelled",
        error: reason
            ? {
                code: "RUN_CANCELLED",
                message: reason,
            }
            : undefined,
    });
}

export async function markRunSucceeded(
    id: string,
    meta?: Record<string, unknown>,
    artifactId?: string,
): Promise<RunRecord | null> {
    return updateRun({
        id,
        status: "succeeded",
        meta,
        artifact_id: artifactId,
        error: null,
    });
}

export async function saveRunArtifact(input: {
    id: string;
    kind: string;
    payload: unknown;
    meta?: Record<string, unknown>;
}): Promise<{ run: RunRecord; artifact: StoredArtifact } | null> {
    const existing = await getRun(input.id);
    if (!existing) {
        return null;
    }

    const artifact = await saveArtifact({
        kind: input.kind,
        payload: input.payload,
        runId: input.id,
    });

    const run = await updateRun({
        id: input.id,
        status: "succeeded",
        artifact_id: artifact.id,
        meta: {
            ...(input.meta || {}),
            artifact_path: artifact.path,
        },
        error: null,
    });

    if (!run) {
        return null;
    }

    return { run, artifact };
}
