import { NextRequest, NextResponse } from "next/server";
import type { EngineError, RunStatus } from "@/lib/contracts/run";
import { isObjectRecord } from "@/lib/contracts/run";
import { readArtifactByPath } from "@/lib/server/artifactStore";
import { enqueueRun, getQueueSnapshot } from "@/lib/server/jobQueue";
import {
    getRun,
    markRunCancelled,
    markRunFailed,
    markRunRunning,
    markRunSucceeded,
    updateRun,
} from "@/lib/server/runStore";
import {
    buildTraceHeaders,
    elapsedMs,
    requestTraceId,
    resolveTraceId,
    telemetryError,
    telemetryInfo,
} from "@/lib/server/telemetry";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const RUN_STATUSES: RunStatus[] = ["queued", "running", "succeeded", "failed", "cancelled"];

function normalizeError(raw: unknown): EngineError | null {
    if (typeof raw === "string" && raw.trim()) {
        return {
            code: "RUN_ERROR",
            message: raw,
        };
    }

    if (!isObjectRecord(raw)) {
        return null;
    }

    const message = typeof raw.message === "string" && raw.message.trim()
        ? raw.message
        : "Run failed";

    return {
        code: typeof raw.code === "string" && raw.code.trim() ? raw.code : "RUN_ERROR",
        message,
        details: raw.details,
    };
}

export async function GET(
    request: NextRequest,
    context: { params: Promise<{ runId: string }> }
) {
    const requestTrace = requestTraceId(request);

    try {
        const startedAt = Date.now();
        const { runId } = await context.params;
        const run = await getRun(runId);
        if (!run) {
            telemetryInfo("api.run.get.not_found", {
                trace_id: requestTrace,
                run_id: runId,
            });
            return NextResponse.json(
                { error: "Run not found." },
                { status: 404, headers: buildTraceHeaders(requestTrace) }
            );
        }

        const traceId = resolveTraceId(run.meta?.trace_id, requestTrace, runId);

        const includeArtifact = new URL(request.url).searchParams.get("include_artifact") === "1";
        if (!includeArtifact) {
            telemetryInfo("api.run.get.success", {
                trace_id: traceId,
                run_id: run.id,
                include_artifact: false,
                status: run.status,
                duration_ms: elapsedMs(startedAt),
            });
            return NextResponse.json(
                { run, queue: getQueueSnapshot() },
                { headers: buildTraceHeaders(traceId, { "Cache-Control": "no-store" }) }
            );
        }

        const artifactPath = isObjectRecord(run.meta) && typeof run.meta.artifact_path === "string"
            ? run.meta.artifact_path
            : null;

        const artifact = artifactPath ? await readArtifactByPath(artifactPath) : null;
        telemetryInfo("api.run.get.success", {
            trace_id: traceId,
            run_id: run.id,
            include_artifact: true,
            has_artifact: Boolean(artifact),
            status: run.status,
            duration_ms: elapsedMs(startedAt),
        });
        return NextResponse.json(
            { run, artifact, queue: getQueueSnapshot() },
            { headers: buildTraceHeaders(traceId, { "Cache-Control": "no-store" }) }
        );
    } catch (error) {
        telemetryError("api.run.get.failed", error, { trace_id: requestTrace });
        const message = error instanceof Error ? error.message : "Failed to load run.";
        return NextResponse.json(
            { error: message },
            { status: 500, headers: buildTraceHeaders(requestTrace) }
        );
    }
}

export async function PATCH(
    request: NextRequest,
    context: { params: Promise<{ runId: string }> }
) {
    const requestTrace = requestTraceId(request);

    try {
        const startedAt = Date.now();
        const { runId } = await context.params;
        const existing = await getRun(runId);
        if (!existing) {
            telemetryInfo("api.run.patch.not_found", {
                trace_id: requestTrace,
                run_id: runId,
            });
            return NextResponse.json(
                { error: "Run not found." },
                { status: 404, headers: buildTraceHeaders(requestTrace) }
            );
        }

        const traceId = resolveTraceId(existing.meta?.trace_id, requestTrace, runId);

        const body: unknown = await request.json();
        if (!isObjectRecord(body)) {
            telemetryInfo("api.run.patch.bad_request", {
                trace_id: traceId,
                run_id: runId,
                reason: "non_object_body",
            });
            return NextResponse.json(
                { error: "Request body must be a JSON object." },
                { status: 400, headers: buildTraceHeaders(traceId) }
            );
        }

        const status = typeof body.status === "string" ? body.status : undefined;
        const action = typeof body.action === "string" ? body.action.toLowerCase() : "";
        const meta = isObjectRecord(body.meta) ? body.meta : undefined;

        if (status && !RUN_STATUSES.includes(status as RunStatus)) {
            telemetryInfo("api.run.patch.bad_request", {
                trace_id: traceId,
                run_id: runId,
                reason: "invalid_status",
                status,
            });
            return NextResponse.json(
                { error: "Invalid run status." },
                { status: 400, headers: buildTraceHeaders(traceId) }
            );
        }

        let updated = null;

        if (action === "enqueue") {
            const queued = await enqueueRun(runId);
            const fresh = await getRun(runId);
            telemetryInfo("api.run.patch.enqueued", {
                trace_id: traceId,
                run_id: runId,
                queue_position: queued.position,
                already_queued: queued.alreadyQueued,
                duration_ms: elapsedMs(startedAt),
            });
            return NextResponse.json(
                { run: fresh, queued, queue: getQueueSnapshot() },
                { headers: buildTraceHeaders(traceId, { "Cache-Control": "no-store" }) }
            );
        }

        if (status === "cancelled") {
            const reason = typeof body.reason === "string" ? body.reason : undefined;
            updated = await markRunCancelled(runId, reason);
        } else if (status === "running") {
            updated = await markRunRunning(runId, {
                ...(meta || {}),
                trace_id: traceId,
            });
        } else if (status === "failed") {
            const err = normalizeError(body.error);
            if (!err) {
                telemetryInfo("api.run.patch.bad_request", {
                    trace_id: traceId,
                    run_id: runId,
                    reason: "missing_error_payload",
                });
                return NextResponse.json(
                    { error: "`error` is required for failed status." },
                    { status: 400, headers: buildTraceHeaders(traceId) }
                );
            }
            updated = await markRunFailed(runId, err, {
                ...(meta || {}),
                trace_id: traceId,
            });
        } else if (status === "succeeded") {
            updated = await markRunSucceeded(runId, {
                ...(meta || {}),
                trace_id: traceId,
            });
        } else {
            updated = await updateRun({
                id: runId,
                status: status as RunStatus | undefined,
                meta: {
                    ...(meta || {}),
                    trace_id: traceId,
                },
            });
        }

        if (!updated) {
            telemetryError("api.run.patch.update_failed", "Run update failed", {
                trace_id: traceId,
                run_id: runId,
            });
            return NextResponse.json(
                { error: "Run update failed." },
                { status: 500, headers: buildTraceHeaders(traceId) }
            );
        }

        telemetryInfo("api.run.patch.success", {
            trace_id: traceId,
            run_id: runId,
            status: updated.status,
            duration_ms: elapsedMs(startedAt),
        });

        return NextResponse.json(
            { run: updated, queue: getQueueSnapshot() },
            { headers: buildTraceHeaders(traceId, { "Cache-Control": "no-store" }) }
        );
    } catch (error) {
        telemetryError("api.run.patch.failed", error, { trace_id: requestTrace });
        const message = error instanceof Error ? error.message : "Failed to update run.";
        return NextResponse.json(
            { error: message },
            { status: 500, headers: buildTraceHeaders(requestTrace) }
        );
    }
}
