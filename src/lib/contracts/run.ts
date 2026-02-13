export type RunKind = "factor_lab" | "signal_construct" | "signal_backtest" | "stock_filter";

export type RunStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled";

export interface EngineError {
    code: string;
    message: string;
    details?: unknown;
}

export interface EngineEnvelope<TResult, TMeta extends Record<string, unknown> = Record<string, unknown>> {
    run_id: string;
    meta: TMeta;
    result: TResult | null;
    error: EngineError | null;
}

export interface ArtifactReference {
    id: string;
    kind: string;
    path: string;
    created_at: string;
}

export interface RunRecord {
    id: string;
    kind: RunKind;
    status: RunStatus;
    created_at: string;
    updated_at: string;
    started_at?: string;
    finished_at?: string;
    request: Record<string, unknown>;
    meta?: Record<string, unknown>;
    artifact_id?: string;
    error?: EngineError;
}

export interface RunListResponse {
    runs: RunRecord[];
    total: number;
}

export interface CreateRunRequest {
    kind: RunKind;
    request: Record<string, unknown>;
    execute?: boolean;
}

export function isObjectRecord(value: unknown): value is Record<string, unknown> {
    return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

export function parseEngineEnvelope<TResult>(value: unknown): EngineEnvelope<TResult> {
    if (!isObjectRecord(value)) {
        throw new Error("Engine response must be a JSON object.");
    }

    const runId = typeof value.run_id === "string" && value.run_id.trim()
        ? value.run_id.trim()
        : "";

    if (!runId) {
        throw new Error("Engine response missing run_id.");
    }

    const meta = isObjectRecord(value.meta) ? value.meta : {};
    const result = ("result" in value ? value.result : null) as TResult | null;
    let error: EngineError | null = null;

    if (value.error !== null && value.error !== undefined) {
        if (typeof value.error === "string") {
            error = {
                code: "ENGINE_ERROR",
                message: value.error,
            };
        } else if (isObjectRecord(value.error)) {
            const code = typeof value.error.code === "string" && value.error.code.trim()
                ? value.error.code
                : "ENGINE_ERROR";
            const message = typeof value.error.message === "string" && value.error.message.trim()
                ? value.error.message
                : "Unknown engine error";
            error = {
                code,
                message,
                details: value.error.details,
            };
        } else {
            error = {
                code: "ENGINE_ERROR",
                message: String(value.error),
            };
        }
    }

    return {
        run_id: runId,
        meta,
        result,
        error,
    };
}
