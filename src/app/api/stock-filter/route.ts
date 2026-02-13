import { NextRequest, NextResponse } from "next/server";
import type { StockFilterPayload, StockFilterMetaResult, StockFilterRunResult } from "@/lib/contracts/screener";
import { executeStockFilterPython } from "@/lib/server/stockFilterPython";
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
    telemetryInfo("api.stock_filter.meta.request", { trace_id: traceId });

    try {
        const startedAt = Date.now();
        const envelope = await executeStockFilterPython<StockFilterMetaResult>({
            _mode: "meta",
            trace_id: traceId,
        } as StockFilterPayload & { trace_id: string });
        if (envelope.error) {
            telemetryError("api.stock_filter.meta.engine_error", envelope.error.message, {
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
            telemetryError("api.stock_filter.meta.empty_result", "Engine returned empty stock filter metadata", {
                trace_id: traceId,
                engine_run_id: envelope.run_id,
                duration_ms: elapsedMs(startedAt),
            });
            return NextResponse.json(
                { error: "Stock filter metadata response did not include result payload.", run_id: envelope.run_id },
                { status: 500, headers: buildTraceHeaders(traceId) }
            );
        }

        telemetryInfo("api.stock_filter.meta.success", {
            trace_id: traceId,
            engine_run_id: envelope.run_id,
            filters: envelope.result.filters.length,
            duration_ms: elapsedMs(startedAt),
        });

        return NextResponse.json(
            { ...envelope.result, run_id: envelope.run_id, run_meta: envelope.meta },
            { headers: buildTraceHeaders(traceId, { "Cache-Control": "no-store" }) }
        );
    } catch (error) {
        telemetryError("api.stock_filter.meta.failed", error, { trace_id: traceId });
        const message = error instanceof Error ? error.message : "Unknown error";
        return NextResponse.json(
            { error: message },
            { status: 500, headers: buildTraceHeaders(traceId) }
        );
    }
}

export async function POST(request: NextRequest) {
    const traceId = requestTraceId(request);
    telemetryInfo("api.stock_filter.run.request", { trace_id: traceId });

    try {
        const startedAt = Date.now();
        const body: unknown = await request.json();
        if (!body || typeof body !== "object" || Array.isArray(body)) {
            telemetryInfo("api.stock_filter.run.bad_request", { trace_id: traceId });
            return NextResponse.json(
                { error: "Request body must be a JSON object." },
                { status: 400, headers: buildTraceHeaders(traceId) }
            );
        }

        const payload = body as StockFilterPayload;
        const envelope = await executeStockFilterPython<StockFilterRunResult>({
            ...payload,
            _mode: "run",
            trace_id: traceId,
        } as StockFilterPayload & { trace_id: string });

        if (envelope.error) {
            telemetryError("api.stock_filter.run.engine_error", envelope.error.message, {
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
            telemetryError("api.stock_filter.run.empty_result", "Engine returned empty stock filter result", {
                trace_id: traceId,
                engine_run_id: envelope.run_id,
                duration_ms: elapsedMs(startedAt),
            });
            return NextResponse.json(
                { error: "Stock filter response did not include result payload.", run_id: envelope.run_id },
                { status: 500, headers: buildTraceHeaders(traceId) }
            );
        }

        telemetryInfo("api.stock_filter.run.success", {
            trace_id: traceId,
            engine_run_id: envelope.run_id,
            rows: envelope.result.rows.length,
            duration_ms: elapsedMs(startedAt),
        });

        return NextResponse.json(
            { ...envelope.result, run_id: envelope.run_id, run_meta: envelope.meta },
            { headers: buildTraceHeaders(traceId, { "Cache-Control": "no-store" }) }
        );
    } catch (error) {
        telemetryError("api.stock_filter.run.failed", error, { trace_id: traceId });
        const message = error instanceof Error ? error.message : "Unknown error";
        return NextResponse.json(
            { error: message },
            { status: 500, headers: buildTraceHeaders(traceId) }
        );
    }
}
