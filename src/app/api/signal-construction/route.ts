import { NextRequest, NextResponse } from "next/server";
import type { SignalConstructionPayload, SignalConstructResult } from "@/lib/contracts/signal";
import { executeSignalPython } from "@/lib/server/signalConstructionPython";
import { getSignalConstructionCatalog } from "@/lib/server/catalogService";
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
    telemetryInfo("api.signal_construct.catalog.request", { trace_id: traceId });

    try {
        const catalog = getSignalConstructionCatalog();
        telemetryInfo("api.signal_construct.catalog.success", {
            trace_id: traceId,
            universes: catalog.universes.length,
            indicators: catalog.indicators.length,
        });
        return NextResponse.json(
            catalog,
            { headers: buildTraceHeaders(traceId, { "Cache-Control": "no-store" }) }
        );
    } catch (error) {
        telemetryError("api.signal_construct.catalog.failed", error, { trace_id: traceId });
        const message = error instanceof Error ? error.message : "Unknown error";
        return NextResponse.json({ error: message }, { status: 500, headers: buildTraceHeaders(traceId) });
    }
}

export async function POST(request: NextRequest) {
    const traceId = requestTraceId(request);
    telemetryInfo("api.signal_construct.request", { trace_id: traceId });

    try {
        const startedAt = Date.now();
        const body: unknown = await request.json();
        if (!body || typeof body !== "object" || Array.isArray(body)) {
            telemetryInfo("api.signal_construct.bad_request", { trace_id: traceId });
            return NextResponse.json(
                { error: "Request body must be a JSON object." },
                { status: 400, headers: buildTraceHeaders(traceId) }
            );
        }

        const payload = body as SignalConstructionPayload;
        const envelope = await executeSignalPython<SignalConstructResult>({
            ...payload,
            _mode: "construct",
            trace_id: traceId,
        } as SignalConstructionPayload & { trace_id: string });

        if (envelope.error) {
            telemetryError("api.signal_construct.engine_error", envelope.error.message, {
                trace_id: traceId,
                engine_run_id: envelope.run_id,
                duration_ms: elapsedMs(startedAt),
            });
            return NextResponse.json(
                { error: envelope.error.message, run_id: envelope.run_id, run_meta: envelope.meta },
                { status: 400, headers: buildTraceHeaders(traceId) }
            );
        }

        if (!envelope.result) {
            telemetryError("api.signal_construct.empty_result", "Engine returned empty signal construction", {
                trace_id: traceId,
                engine_run_id: envelope.run_id,
                duration_ms: elapsedMs(startedAt),
            });
            return NextResponse.json(
                { error: "Signal construction response did not include result payload.", run_id: envelope.run_id },
                { status: 500, headers: buildTraceHeaders(traceId) }
            );
        }

        telemetryInfo("api.signal_construct.success", {
            trace_id: traceId,
            engine_run_id: envelope.run_id,
            symbols: envelope.result.signals.length,
            duration_ms: elapsedMs(startedAt),
        });

        return NextResponse.json(
            { ...envelope.result, run_id: envelope.run_id, run_meta: envelope.meta },
            {
                headers: buildTraceHeaders(traceId, {
                    "Cache-Control": "no-store, max-age=0",
                }),
            }
        );
    } catch (error) {
        telemetryError("api.signal_construct.failed", error, { trace_id: traceId });
        const message = error instanceof Error ? error.message : "Unknown error";
        return NextResponse.json(
            { error: message },
            {
                status: 500,
                headers: buildTraceHeaders(traceId, {
                    "Cache-Control": "no-store",
                }),
            }
        );
    }
}
