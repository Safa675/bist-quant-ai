import { NextRequest, NextResponse } from "next/server";
import type { RunKind, RunStatus } from "@/lib/contracts/run";
import { isObjectRecord } from "@/lib/contracts/run";
import { createRun, listRuns } from "@/lib/server/runStore";
import { executeRunNow } from "@/lib/server/runExecution";
import { enqueueRun, getQueueSnapshot } from "@/lib/server/jobQueue";
import {
    buildTraceHeaders,
    elapsedMs,
    requestTraceId,
    telemetryError,
    telemetryInfo,
} from "@/lib/server/telemetry";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 300;

const RUN_KINDS: RunKind[] = ["factor_lab", "signal_construct", "signal_backtest", "stock_filter"];
const RUN_STATUSES: RunStatus[] = ["queued", "running", "succeeded", "failed", "cancelled"];

function parsePositiveInt(raw: string | null, fallback: number): number {
    if (!raw) return fallback;
    const parsed = Number.parseInt(raw, 10);
    if (!Number.isFinite(parsed) || parsed < 0) return fallback;
    return parsed;
}

async function parseJsonObjectBody(request: NextRequest): Promise<Record<string, unknown> | null> {
    try {
        const body: unknown = await request.json();
        return isObjectRecord(body) ? body : null;
    } catch {
        return null;
    }
}

export async function GET(request: NextRequest) {
    const traceId = requestTraceId(request);
    telemetryInfo("api.runs.list.request", { trace_id: traceId });

    try {
        const startedAt = Date.now();
        const url = new URL(request.url);
        const kindParam = url.searchParams.get("kind");
        const statusParam = url.searchParams.get("status");

        const kind = RUN_KINDS.includes(kindParam as RunKind)
            ? (kindParam as RunKind)
            : undefined;
        const status = RUN_STATUSES.includes(statusParam as RunStatus)
            ? (statusParam as RunStatus)
            : undefined;

        const limit = Math.max(1, Math.min(parsePositiveInt(url.searchParams.get("limit"), 100), 500));
        const offset = parsePositiveInt(url.searchParams.get("offset"), 0);

        const { runs, total } = await listRuns({ kind, status, limit, offset });
        const queue = getQueueSnapshot();

        telemetryInfo("api.runs.list.success", {
            trace_id: traceId,
            total,
            returned: runs.length,
            kind: kind || null,
            status: status || null,
            duration_ms: elapsedMs(startedAt),
        });

        return NextResponse.json(
            { runs, total, queue },
            { headers: buildTraceHeaders(traceId, { "Cache-Control": "no-store" }) }
        );
    } catch (error) {
        telemetryError("api.runs.list.failed", error, { trace_id: traceId });
        const message = error instanceof Error ? error.message : "Failed to list runs.";
        return NextResponse.json(
            { error: message },
            { status: 500, headers: buildTraceHeaders(traceId) }
        );
    }
}

export async function POST(request: NextRequest) {
    const traceId = requestTraceId(request);
    telemetryInfo("api.runs.create.request", { trace_id: traceId });

    try {
        const startedAt = Date.now();
        const body = await parseJsonObjectBody(request);
        if (!body) {
            telemetryInfo("api.runs.create.bad_request", {
                trace_id: traceId,
                reason: "invalid_json_or_non_object",
            });
            return NextResponse.json(
                { error: "Request body must be a JSON object." },
                { status: 400, headers: buildTraceHeaders(traceId) }
            );
        }

        const kind = typeof body.kind === "string" ? body.kind : "";
        const requestPayload = isObjectRecord(body.request) ? body.request : null;
        const execute = body.execute === true;
        const blocking = body.blocking === true;
        const forceBlocking = process.env.RUNS_FORCE_BLOCKING === "1" || process.env.VERCEL === "1";

        if (!RUN_KINDS.includes(kind as RunKind)) {
            telemetryInfo("api.runs.create.bad_request", { trace_id: traceId, reason: "invalid_kind", kind });
            return NextResponse.json(
                { error: "Invalid run kind." },
                { status: 400, headers: buildTraceHeaders(traceId) }
            );
        }
        if (!requestPayload) {
            telemetryInfo("api.runs.create.bad_request", { trace_id: traceId, reason: "missing_request_payload" });
            return NextResponse.json(
                { error: "`request` must be a JSON object." },
                { status: 400, headers: buildTraceHeaders(traceId) }
            );
        }

        const run = await createRun({
            kind: kind as RunKind,
            request: {
                ...requestPayload,
                trace_id: traceId,
            },
            status: "queued",
            meta: {
                source: "api/runs",
                execute,
                blocking,
                trace_id: traceId,
            },
        });

        telemetryInfo("api.runs.create.created", {
            trace_id: traceId,
            run_id: run.id,
            kind: run.kind,
            execute,
            blocking,
            force_blocking: forceBlocking,
        });

        if (!execute) {
            return NextResponse.json(
                { run },
                { status: 201, headers: buildTraceHeaders(traceId, { "Cache-Control": "no-store" }) }
            );
        }

        if (execute && (blocking || forceBlocking)) {
            if (!blocking && forceBlocking) {
                telemetryInfo("api.runs.create.force_blocking", {
                    trace_id: traceId,
                    run_id: run.id,
                    kind: run.kind,
                });
            }
            const outcome = await executeRunNow(run);
            const statusCode = outcome.run.status === "failed" ? 400 : 200;

            telemetryInfo("api.runs.create.blocking_complete", {
                trace_id: traceId,
                run_id: run.id,
                final_status: outcome.run.status,
                artifact_id: outcome.artifact?.id,
                duration_ms: elapsedMs(startedAt),
            });

            return NextResponse.json(
                {
                    run: outcome.run,
                    envelope: outcome.envelope,
                    artifact: outcome.artifact || null,
                },
                {
                    status: statusCode,
                    headers: buildTraceHeaders(traceId, { "Cache-Control": "no-store" }),
                }
            );
        }

        const queued = await enqueueRun(run.id);
        const queue = getQueueSnapshot();

        telemetryInfo("api.runs.create.queued", {
            trace_id: traceId,
            run_id: run.id,
            queue_position: queued.position,
            already_queued: queued.alreadyQueued,
            pending: queue.pending,
            active: queue.active,
            duration_ms: elapsedMs(startedAt),
        });

        return NextResponse.json(
            {
                run,
                queued,
                queue,
            },
            { status: 202, headers: buildTraceHeaders(traceId, { "Cache-Control": "no-store" }) }
        );
    } catch (error) {
        telemetryError("api.runs.create.failed", error, { trace_id: traceId });
        const message = error instanceof Error ? error.message : "Failed to create run.";
        return NextResponse.json(
            { error: message },
            { status: 500, headers: buildTraceHeaders(traceId) }
        );
    }
}
