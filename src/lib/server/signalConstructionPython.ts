import { spawn } from "child_process";
import path from "path";
import type { SignalConstructionPayload } from "@/lib/contracts/signal";
import { parseEngineEnvelope, type EngineEnvelope } from "@/lib/contracts/run";
import { elapsedMs, resolveTraceId, telemetryError, telemetryInfo } from "@/lib/server/telemetry";

const PYTHON_SCRIPT = path.resolve(process.cwd(), "dashboard", "signal_construction_api.py");
const REMOTE_ENGINE_URL = (process.env.SIGNAL_ENGINE_URL || "").trim();

const LOCAL_SIGNAL_API_PATHS = [
    "/py-api/api/signal_construction",
    "/api/signal_construction",
    "/api/index.py/api/signal_construction",
];

function asErrorMessage(err: unknown): string {
    return err instanceof Error ? err.message : String(err);
}

function getBaseUrl(): string {
    const publicAppUrl = (process.env.NEXT_PUBLIC_APP_URL || "").trim();
    if (publicAppUrl) {
        return publicAppUrl.replace(/\/+$/, "");
    }

    const vercelUrl = (process.env.VERCEL_URL || "").trim();
    if (vercelUrl) {
        return `https://${vercelUrl}`;
    }

    return "http://localhost:3000";
}

function buildRemoteCandidates(): string[] {
    const candidates: string[] = [];
    if (REMOTE_ENGINE_URL) {
        candidates.push(REMOTE_ENGINE_URL);
    }

    const baseUrl = getBaseUrl();
    for (const apiPath of LOCAL_SIGNAL_API_PATHS) {
        candidates.push(`${baseUrl}${apiPath}`);
    }

    return Array.from(new Set(candidates.map((url) => url.trim()).filter(Boolean)));
}

async function executeRemoteSignalEngine<TResult>(
    url: string,
    payload: SignalConstructionPayload
): Promise<EngineEnvelope<TResult>> {
    const response = await fetch(url, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
        cache: "no-store",
    });

    const rawText = await response.text();
    let parsed: unknown = null;
    try {
        parsed = JSON.parse(rawText);
    } catch {
        const snippet = rawText.slice(0, 180).replace(/\s+/g, " ").trim();
        throw new Error(
            `Remote signal engine returned non-JSON response (${response.status}) from ${url}. ` +
            `Body starts with: ${snippet || "(empty)"}`
        );
    }

    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
        throw new Error(`Remote signal engine response must be a JSON object (${url}).`);
    }

    const result = parsed as Record<string, unknown>;
    if (!response.ok) {
        const remoteError = typeof result.error === "string"
            ? result.error
            : `Remote signal engine failed (${response.status}) at ${url}`;
        throw new Error(remoteError);
    }
    return parseEngineEnvelope<TResult>(result);
}

function parseJsonFromStdout(stdout: string): Record<string, unknown> {
    const trimmed = stdout.trim();
    if (!trimmed) {
        throw new Error("Python script returned empty output.");
    }

    try {
        const parsed: unknown = JSON.parse(trimmed);
        if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
            return parsed as Record<string, unknown>;
        }
    } catch {
        // Ignore; try line-by-line fallback.
    }

    const lines = trimmed
        .split("\n")
        .map((line) => line.trim())
        .filter(Boolean)
        .reverse();

    for (const line of lines) {
        try {
            const parsed: unknown = JSON.parse(line);
            if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
                return parsed as Record<string, unknown>;
            }
        } catch {
            continue;
        }
    }

    throw new Error(`Failed to parse Python output as JSON: ${trimmed}`);
}

async function executeLocal<TResult>(payload: SignalConstructionPayload): Promise<EngineEnvelope<TResult>> {
    return new Promise((resolve, reject) => {
        const child = spawn("python3", [PYTHON_SCRIPT], {
            cwd: path.dirname(PYTHON_SCRIPT),
        });

        let stdout = "";
        let stderr = "";

        const timeoutHandle = setTimeout(() => {
            child.kill("SIGKILL");
            reject(new Error("Signal API timeout (180s)."));
        }, 180_000);

        child.stdout.on("data", (data: Buffer) => {
            stdout += data.toString();
        });

        child.stderr.on("data", (data: Buffer) => {
            stderr += data.toString();
        });

        child.on("error", (err) => {
            clearTimeout(timeoutHandle);
            reject(new Error(`Failed to spawn Python process: ${err.message}`));
        });

        child.on("close", (code) => {
            clearTimeout(timeoutHandle);
            if (code !== 0) {
                reject(new Error(`Python process failed (${code}): ${stderr || "unknown error"}`));
                return;
            }

            try {
                resolve(parseEngineEnvelope<TResult>(parseJsonFromStdout(stdout)));
            } catch (err) {
                const message = err instanceof Error ? err.message : String(err);
                reject(new Error(`${message}\nStderr: ${stderr || "(none)"}`));
            }
        });

        try {
            child.stdin.write(JSON.stringify(payload));
            child.stdin.end();
        } catch (err) {
            clearTimeout(timeoutHandle);
            reject(new Error(`Failed to send payload to Python process: ${String(err)}`));
        }
    });
}

export async function executeSignalPython<TResult = Record<string, unknown>>(
    payload: SignalConstructionPayload
): Promise<EngineEnvelope<TResult>> {
    const payloadRecord = payload as Record<string, unknown>;
    const traceId = resolveTraceId(payloadRecord.trace_id, payloadRecord.run_id);
    const requestedRunId = typeof payloadRecord.run_id === "string" ? payloadRecord.run_id : undefined;
    const remoteCandidates = buildRemoteCandidates();

    telemetryInfo("engine.signal.request", {
        trace_id: traceId,
        run_id: requestedRunId,
        mode: payload._mode || "construct",
        remote_candidates: remoteCandidates.length,
    });

    const localStartedAt = Date.now();
    try {
        const envelope = await executeLocal<TResult>(payload);
        telemetryInfo("engine.signal.local.success", {
            trace_id: traceId,
            run_id: requestedRunId || envelope.run_id,
            duration_ms: elapsedMs(localStartedAt),
        });
        return envelope;
    } catch (localErr) {
        telemetryError("engine.signal.local.failed", localErr, {
            trace_id: traceId,
            run_id: requestedRunId,
            duration_ms: elapsedMs(localStartedAt),
        });

        const remoteErrors: string[] = [];
        for (const url of remoteCandidates) {
            const remoteStartedAt = Date.now();
            try {
                const envelope = await executeRemoteSignalEngine<TResult>(url, payload);
                telemetryInfo("engine.signal.remote.success", {
                    trace_id: traceId,
                    run_id: requestedRunId || envelope.run_id,
                    url,
                    duration_ms: elapsedMs(remoteStartedAt),
                });
                return envelope;
            } catch (err) {
                remoteErrors.push(`${url} -> ${asErrorMessage(err)}`);
                telemetryError("engine.signal.remote.failed", err, {
                    trace_id: traceId,
                    run_id: requestedRunId,
                    url,
                    duration_ms: elapsedMs(remoteStartedAt),
                });
            }
        }

        const details = [`local python -> ${asErrorMessage(localErr)}`, ...remoteErrors]
            .map((entry, idx) => `${idx + 1}. ${entry}`)
            .join("\n");
        telemetryError("engine.signal.failed", localErr, {
            trace_id: traceId,
            run_id: requestedRunId,
            attempts: 1 + remoteCandidates.length,
        });
        throw new Error(`Signal engine failed for all execution paths.\n${details}`);
    }
}
