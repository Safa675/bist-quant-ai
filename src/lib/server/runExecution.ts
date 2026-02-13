import {
    type EngineEnvelope,
    type EngineError,
    type RunKind,
    type RunRecord,
    isObjectRecord,
} from "@/lib/contracts/run";
import type { FactorLabPayload } from "@/lib/contracts/factor";
import type { SignalConstructionPayload } from "@/lib/contracts/signal";
import type { StockFilterPayload } from "@/lib/contracts/screener";
import { executeFactorLabPython } from "@/lib/server/factorLabPython";
import { executeSignalPython } from "@/lib/server/signalConstructionPython";
import { executeStockFilterPython } from "@/lib/server/stockFilterPython";
import {
    markRunFailed,
    markRunRunning,
    saveRunArtifact,
} from "@/lib/server/runStore";
import type { StoredArtifact } from "@/lib/server/artifactStore";
import { telemetryError, telemetryInfo, resolveTraceId } from "@/lib/server/telemetry";

export interface RunExecutionOutcome {
    run: RunRecord;
    envelope: EngineEnvelope<Record<string, unknown>>;
    artifact?: StoredArtifact;
}

function requestObject(input: unknown): Record<string, unknown> {
    if (!isObjectRecord(input)) {
        return {};
    }
    return { ...input };
}

function toEngineError(error: unknown): EngineError {
    if (isObjectRecord(error)) {
        const code = typeof error.code === "string" && error.code.trim() ? error.code : "RUN_EXECUTION_FAILED";
        const message = typeof error.message === "string" && error.message.trim()
            ? error.message
            : "Run execution failed";
        return {
            code,
            message,
            details: error.details,
        };
    }

    const message = error instanceof Error ? error.message : String(error);
    return {
        code: "RUN_EXECUTION_FAILED",
        message,
    };
}

async function dispatchRun(kind: RunKind, request: Record<string, unknown>, runId: string) {
    const traceId = resolveTraceId(request.trace_id, runId);
    const basePayload = { ...request, run_id: runId, trace_id: traceId };

    if (kind === "factor_lab") {
        return executeFactorLabPython<Record<string, unknown>>(
            basePayload as FactorLabPayload & { run_id: string }
        );
    }

    if (kind === "signal_construct") {
        return executeSignalPython<Record<string, unknown>>(
            { ...basePayload, _mode: "construct" } as SignalConstructionPayload & { run_id: string }
        );
    }

    if (kind === "signal_backtest") {
        return executeSignalPython<Record<string, unknown>>(
            { ...basePayload, _mode: "backtest" } as SignalConstructionPayload & { run_id: string }
        );
    }

    if (kind === "stock_filter") {
        return executeStockFilterPython<Record<string, unknown>>(
            { ...basePayload, _mode: "run" } as StockFilterPayload & { run_id: string }
        );
    }

    throw new Error(`Unsupported run kind: ${kind}`);
}

export async function executeRunNow(run: RunRecord): Promise<RunExecutionOutcome> {
    const traceId = resolveTraceId(run.meta?.trace_id, run.id);
    const running = await markRunRunning(run.id, {
        executor: "sync_api",
        trace_id: traceId,
    });
    const activeRun = running || run;

    telemetryInfo("run.execution.started", {
        trace_id: traceId,
        run_id: activeRun.id,
        kind: activeRun.kind,
    });

    try {
        const request = requestObject(activeRun.request);
        request.trace_id = traceId;
        const envelope = await dispatchRun(activeRun.kind, request, activeRun.id);

        if (envelope.error) {
            telemetryError("run.execution.engine_error", envelope.error.message, {
                trace_id: traceId,
                run_id: activeRun.id,
                kind: activeRun.kind,
                engine_run_id: envelope.run_id,
            });
            const failed = await markRunFailed(activeRun.id, envelope.error, {
                engine_meta: envelope.meta,
                engine_run_id: envelope.run_id,
                trace_id: traceId,
            });
            return {
                run: failed || {
                    ...activeRun,
                    status: "failed",
                    error: envelope.error,
                },
                envelope,
            };
        }

        if (!envelope.result) {
            const err: EngineError = {
                code: "RUN_EMPTY_RESULT",
                message: "Engine returned no result payload.",
            };
            telemetryError("run.execution.empty_result", err.message, {
                trace_id: traceId,
                run_id: activeRun.id,
                kind: activeRun.kind,
                engine_run_id: envelope.run_id,
            });
            const failed = await markRunFailed(activeRun.id, err, {
                engine_meta: envelope.meta,
                engine_run_id: envelope.run_id,
                trace_id: traceId,
            });
            return {
                run: failed || {
                    ...activeRun,
                    status: "failed",
                    error: err,
                },
                envelope: {
                    ...envelope,
                    error: err,
                },
            };
        }

        const saved = await saveRunArtifact({
            id: activeRun.id,
            kind: activeRun.kind,
            payload: envelope.result,
            meta: {
                engine_meta: envelope.meta,
                engine_run_id: envelope.run_id,
                trace_id: traceId,
            },
        });

        if (!saved) {
            throw new Error(`Run not found while saving artifact: ${activeRun.id}`);
        }

        telemetryInfo("run.execution.succeeded", {
            trace_id: traceId,
            run_id: activeRun.id,
            kind: activeRun.kind,
            engine_run_id: envelope.run_id,
            artifact_id: saved.artifact.id,
        });

        return {
            run: saved.run,
            envelope,
            artifact: saved.artifact,
        };
    } catch (error) {
        const engineError = toEngineError(error);
        telemetryError("run.execution.failed", engineError.message, {
            trace_id: traceId,
            run_id: activeRun.id,
            kind: activeRun.kind,
        });
        const failed = await markRunFailed(activeRun.id, engineError);

        return {
            run: failed || {
                ...activeRun,
                status: "failed",
                error: engineError,
            },
            envelope: {
                run_id: activeRun.id,
                meta: {},
                result: null,
                error: engineError,
            },
        };
    }
}
