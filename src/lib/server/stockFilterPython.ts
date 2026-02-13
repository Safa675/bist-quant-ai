import { spawn } from "child_process";
import path from "path";
import type { StockFilterPayload } from "@/lib/contracts/screener";
import { parseEngineEnvelope, type EngineEnvelope } from "@/lib/contracts/run";
import { elapsedMs, resolveTraceId, telemetryError, telemetryInfo } from "@/lib/server/telemetry";

const PYTHON_SCRIPT = path.resolve(process.cwd(), "dashboard", "stock_filter_api.py");
const REMOTE_ENGINE_URL = (process.env.STOCK_FILTER_ENGINE_URL || "").trim();

const LOCAL_STOCK_FILTER_API_PATHS = [
    "/py-api/api/stock_filter",
    "/api/stock_filter",
    "/api/index.py/api/stock_filter",
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
    for (const apiPath of LOCAL_STOCK_FILTER_API_PATHS) {
        candidates.push(`${baseUrl}${apiPath}`);
    }

    return Array.from(new Set(candidates.map((url) => url.trim()).filter(Boolean)));
}

function parseJsonFromStdout(stdout: string): Record<string, unknown> {
    const trimmed = stdout.trim();
    if (!trimmed) {
        throw new Error("Python stock filter returned empty output.");
    }

    try {
        const parsed: unknown = JSON.parse(trimmed);
        if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
            return parsed as Record<string, unknown>;
        }
    } catch {
        // Ignore and fallback to line parser.
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

async function executeRemote<TResult>(
    url: string,
    payload: StockFilterPayload
): Promise<EngineEnvelope<TResult>> {
    const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
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
            `Remote stock filter returned non-JSON response (${response.status}) from ${url}. ` +
            `Body starts with: ${snippet || "(empty)"}`
        );
    }

    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
        throw new Error(`Remote stock filter response must be a JSON object (${url}).`);
    }

    const result = parsed as Record<string, unknown>;
    if (!response.ok) {
        const msg = typeof result.error === "string"
            ? result.error
            : `Remote stock filter failed (${response.status}) at ${url}`;
        throw new Error(msg);
    }

    return parseEngineEnvelope<TResult>(result);
}

async function executeLocal<TResult>(payload: StockFilterPayload): Promise<EngineEnvelope<TResult>> {
    return new Promise((resolve, reject) => {
        const child = spawn("python3", [PYTHON_SCRIPT], {
            cwd: path.dirname(PYTHON_SCRIPT),
        });

        let stdout = "";
        let stderr = "";

        const timeout = setTimeout(() => {
            child.kill("SIGKILL");
            reject(new Error("Stock filter timeout (90s)."));
        }, 90_000);

        child.stdout.on("data", (chunk: Buffer) => {
            stdout += chunk.toString();
        });

        child.stderr.on("data", (chunk: Buffer) => {
            stderr += chunk.toString();
        });

        child.on("error", (err) => {
            clearTimeout(timeout);
            reject(new Error(`Failed to spawn Python process: ${err.message}`));
        });

        child.on("close", (code) => {
            clearTimeout(timeout);
            if (code !== 0) {
                reject(new Error(`Python process failed (${code}): ${stderr || "unknown error"}`));
                return;
            }

            try {
                resolve(parseEngineEnvelope<TResult>(parseJsonFromStdout(stdout)));
            } catch (err) {
                const msg = err instanceof Error ? err.message : String(err);
                reject(new Error(`${msg}\nStderr: ${stderr || "(none)"}`));
            }
        });

        child.stdin.write(JSON.stringify(payload));
        child.stdin.end();
    });
}

export async function executeStockFilterPython<TResult = Record<string, unknown>>(
    payload: StockFilterPayload
): Promise<EngineEnvelope<TResult>> {
    const payloadRecord = payload as Record<string, unknown>;
    const traceId = resolveTraceId(payloadRecord.trace_id, payloadRecord.run_id);
    const requestedRunId = typeof payloadRecord.run_id === "string" ? payloadRecord.run_id : undefined;
    const remoteCandidates = buildRemoteCandidates();

    telemetryInfo("engine.stock_filter.request", {
        trace_id: traceId,
        run_id: requestedRunId,
        mode: payload._mode || "run",
        remote_candidates: remoteCandidates.length,
    });

    // Always prefer local python for consistency with local codebase.
    const localStartedAt = Date.now();
    try {
        const envelope = await executeLocal<TResult>(payload);
        telemetryInfo("engine.stock_filter.local.success", {
            trace_id: traceId,
            run_id: requestedRunId || envelope.run_id,
            duration_ms: elapsedMs(localStartedAt),
        });
        return envelope;
    } catch (localErr) {
        telemetryError("engine.stock_filter.local.failed", localErr, {
            trace_id: traceId,
            run_id: requestedRunId,
            duration_ms: elapsedMs(localStartedAt),
        });

        const remoteErrors: string[] = [];
        for (const url of remoteCandidates) {
            const remoteStartedAt = Date.now();
            try {
                const envelope = await executeRemote<TResult>(url, payload);
                telemetryInfo("engine.stock_filter.remote.success", {
                    trace_id: traceId,
                    run_id: requestedRunId || envelope.run_id,
                    url,
                    duration_ms: elapsedMs(remoteStartedAt),
                });
                return envelope;
            } catch (err) {
                remoteErrors.push(`${url} -> ${asErrorMessage(err)}`);
                telemetryError("engine.stock_filter.remote.failed", err, {
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
        telemetryError("engine.stock_filter.failed", localErr, {
            trace_id: traceId,
            run_id: requestedRunId,
            attempts: 1 + remoteCandidates.length,
        });
        throw new Error(`Stock filter engine failed for all execution paths.\n${details}`);
    }
}
