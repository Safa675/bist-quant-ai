import { NextRequest, NextResponse } from "next/server";
import type { FactorLabBacktestResult, FactorLabPayload } from "@/lib/contracts/factor";
import { executeFactorLabPython } from "@/lib/server/factorLabPython";
import { createRun, markRunFailed, saveRunArtifact } from "@/lib/server/runStore";
import { getFactorCatalog } from "@/lib/server/catalogService";
import {
    buildTraceHeaders,
    elapsedMs,
    requestTraceId,
    telemetryError,
    telemetryInfo,
} from "@/lib/server/telemetry";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
    const traceId = requestTraceId(request);
    telemetryInfo("api.factor_lab.catalog.request", { trace_id: traceId });

    try {
        const startedAt = Date.now();
        const catalog = await getFactorCatalog();
        telemetryInfo("api.factor_lab.catalog.success", {
            trace_id: traceId,
            source: catalog.source,
            factor_count: catalog.factors.length,
            duration_ms: elapsedMs(startedAt),
        });

        return NextResponse.json(
            {
                factors: catalog.factors,
                default_portfolio_options: catalog.default_portfolio_options,
                catalog_source: catalog.source,
                catalog_error: catalog.error,
            },
            { headers: buildTraceHeaders(traceId, { "Cache-Control": "no-store" }) }
        );
    } catch (error) {
        telemetryError("api.factor_lab.catalog.failed", error, { trace_id: traceId });
        const message = error instanceof Error ? error.message : "Unknown error";
        return NextResponse.json(
            { error: message },
            { status: 500, headers: buildTraceHeaders(traceId) }
        );
    }
}

export async function POST(request: NextRequest) {
    const traceId = requestTraceId(request);
    telemetryInfo("api.factor_lab.run.request", { trace_id: traceId });

    try {
        const startedAt = Date.now();
        const body: unknown = await request.json();
        if (!body || typeof body !== "object" || Array.isArray(body)) {
            telemetryInfo("api.factor_lab.run.bad_request", { trace_id: traceId });
            return NextResponse.json(
                { error: "Request body must be a JSON object." },
                { status: 400, headers: buildTraceHeaders(traceId) }
            );
        }

        const payload = body as FactorLabPayload;
        const run = await createRun({
            kind: "factor_lab",
            request: { ...(payload as Record<string, unknown>), _mode: "run", trace_id: traceId },
            status: "running",
            meta: {
                source: "api/factor-lab",
                trace_id: traceId,
            },
        });

        telemetryInfo("api.factor_lab.run.created", {
            trace_id: traceId,
            run_id: run.id,
        });

        const envelope = await executeFactorLabPython<FactorLabBacktestResult>({
            ...payload,
            _mode: "run",
            run_id: run.id,
            trace_id: traceId,
        } as FactorLabPayload & { run_id: string; trace_id: string });

        if (envelope.error) {
            await markRunFailed(run.id, envelope.error, {
                engine_meta: envelope.meta,
                engine_run_id: envelope.run_id,
                trace_id: traceId,
            });
            telemetryError("api.factor_lab.run.engine_error", envelope.error.message, {
                trace_id: traceId,
                run_id: run.id,
                engine_run_id: envelope.run_id,
                duration_ms: elapsedMs(startedAt),
            });
            return NextResponse.json(
                {
                    error: envelope.error.message,
                    run_id: run.id,
                    run_meta: envelope.meta,
                },
                { status: 400, headers: buildTraceHeaders(traceId) }
            );
        }

        if (!envelope.result) {
            await markRunFailed(run.id, {
                code: "RUN_EMPTY_RESULT",
                message: "Factor run response did not include result payload.",
            }, {
                engine_meta: envelope.meta,
                engine_run_id: envelope.run_id,
                trace_id: traceId,
            });
            telemetryError("api.factor_lab.run.empty_result", "Engine returned empty factor result", {
                trace_id: traceId,
                run_id: run.id,
                engine_run_id: envelope.run_id,
                duration_ms: elapsedMs(startedAt),
            });
            return NextResponse.json(
                { error: "Factor run response did not include result payload.", run_id: run.id },
                { status: 500, headers: buildTraceHeaders(traceId) }
            );
        }

        const saved = await saveRunArtifact({
            id: run.id,
            kind: "factor_lab",
            payload: envelope.result,
            meta: {
                engine_meta: envelope.meta,
                engine_run_id: envelope.run_id,
                trace_id: traceId,
            },
        });

        telemetryInfo("api.factor_lab.run.success", {
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
        telemetryError("api.factor_lab.run.failed", error, { trace_id: traceId });
        const message = error instanceof Error ? error.message : "Unknown error";
        return NextResponse.json(
            { error: message },
            { status: 500, headers: buildTraceHeaders(traceId) }
        );
    }
}
