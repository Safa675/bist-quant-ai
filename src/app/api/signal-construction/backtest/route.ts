import { NextRequest, NextResponse } from "next/server";
import type { SignalBacktestResult, SignalConstructionPayload } from "@/lib/contracts/signal";
import { executeSignalPython } from "@/lib/server/signalConstructionPython";
import { createRun, markRunFailed, saveRunArtifact } from "@/lib/server/runStore";
import {
    buildTraceHeaders,
    elapsedMs,
    requestTraceId,
    telemetryError,
    telemetryInfo,
} from "@/lib/server/telemetry";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(request: NextRequest) {
    const traceId = requestTraceId(request);
    telemetryInfo("api.signal_backtest.request", { trace_id: traceId });

    try {
        const startedAt = Date.now();
        const body: unknown = await request.json();
        if (!body || typeof body !== "object" || Array.isArray(body)) {
            telemetryInfo("api.signal_backtest.bad_request", { trace_id: traceId });
            return NextResponse.json(
                { error: "Request body must be a JSON object." },
                { status: 400, headers: buildTraceHeaders(traceId) }
            );
        }

        const payload = body as SignalConstructionPayload;
        const run = await createRun({
            kind: "signal_backtest",
            request: { ...(payload as Record<string, unknown>), _mode: "backtest", trace_id: traceId },
            status: "running",
            meta: {
                source: "api/signal-construction/backtest",
                trace_id: traceId,
            },
        });

        telemetryInfo("api.signal_backtest.run.created", {
            trace_id: traceId,
            run_id: run.id,
        });

        const envelope = await executeSignalPython<SignalBacktestResult>({
            ...payload,
            _mode: "backtest",
            run_id: run.id,
            trace_id: traceId,
        } as SignalConstructionPayload & { run_id: string; trace_id: string });

        if (envelope.error) {
            await markRunFailed(run.id, envelope.error, {
                engine_meta: envelope.meta,
                engine_run_id: envelope.run_id,
                trace_id: traceId,
            });
            telemetryError("api.signal_backtest.engine_error", envelope.error.message, {
                trace_id: traceId,
                run_id: run.id,
                engine_run_id: envelope.run_id,
                duration_ms: elapsedMs(startedAt),
            });
            return NextResponse.json(
                { error: envelope.error.message, run_id: run.id, run_meta: envelope.meta },
                { status: 400, headers: buildTraceHeaders(traceId) }
            );
        }

        if (!envelope.result) {
            await markRunFailed(run.id, {
                code: "RUN_EMPTY_RESULT",
                message: "Signal backtest response did not include result payload.",
            }, {
                engine_meta: envelope.meta,
                engine_run_id: envelope.run_id,
                trace_id: traceId,
            });
            telemetryError("api.signal_backtest.empty_result", "Engine returned empty signal backtest", {
                trace_id: traceId,
                run_id: run.id,
                engine_run_id: envelope.run_id,
                duration_ms: elapsedMs(startedAt),
            });
            return NextResponse.json(
                { error: "Signal backtest response did not include result payload.", run_id: run.id },
                { status: 500, headers: buildTraceHeaders(traceId) }
            );
        }

        const saved = await saveRunArtifact({
            id: run.id,
            kind: "signal_backtest",
            payload: envelope.result,
            meta: {
                engine_meta: envelope.meta,
                engine_run_id: envelope.run_id,
                trace_id: traceId,
            },
        });

        telemetryInfo("api.signal_backtest.success", {
            trace_id: traceId,
            run_id: run.id,
            engine_run_id: envelope.run_id,
            artifact_id: saved?.artifact.id,
            duration_ms: elapsedMs(startedAt),
        });

        return NextResponse.json(
            {
                ...envelope.result,
                run_id: run.id,
                run_meta: envelope.meta,
                artifact_id: saved?.artifact.id,
            },
            { headers: buildTraceHeaders(traceId, { "Cache-Control": "no-store" }) }
        );
    } catch (error) {
        telemetryError("api.signal_backtest.failed", error, { trace_id: traceId });
        const message = error instanceof Error ? error.message : "Unknown error";
        return NextResponse.json(
            { error: message },
            { status: 500, headers: buildTraceHeaders(traceId) }
        );
    }
}
