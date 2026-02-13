import { spawn } from "child_process";
import path from "path";

const PYTHON_SCRIPT = path.resolve(process.cwd(), "dashboard", "signal_construction_api.py");
const REMOTE_ENGINE_URL = (process.env.SIGNAL_ENGINE_URL || "").trim();

const LOCAL_SIGNAL_API_PATHS = [
    "/py-api/api/signal_construction",
    "/api/signal_construction",
    "/api/index.py/api/signal_construction",
];

export interface SignalConstructionPayload {
    universe?: string;
    symbols?: string[] | string;
    period?: string;
    interval?: string;
    max_symbols?: number;
    top_n?: number;
    buy_threshold?: number;
    sell_threshold?: number;
    indicators?: Record<string, { enabled?: boolean; params?: Record<string, number> }>;
    _mode?: "construct" | "backtest";
}

function asErrorMessage(err: unknown): string {
    return err instanceof Error ? err.message : String(err);
}

function getBaseUrl(): string {
    const vercelUrl = (process.env.VERCEL_URL || "").trim();
    if (vercelUrl) {
        return `https://${vercelUrl}`;
    }

    const publicAppUrl = (process.env.NEXT_PUBLIC_APP_URL || "").trim();
    if (publicAppUrl) {
        return publicAppUrl.replace(/\/+$/, "");
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

async function executeRemoteSignalEngine(
    url: string,
    payload: SignalConstructionPayload
): Promise<Record<string, unknown>> {
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
    return result;
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

async function executeLocal(payload: SignalConstructionPayload): Promise<Record<string, unknown>> {
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
                resolve(parseJsonFromStdout(stdout));
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

export async function executeSignalPython(payload: SignalConstructionPayload): Promise<Record<string, unknown>> {
    const remoteCandidates = buildRemoteCandidates();

    // If a dedicated remote engine is configured, try it first.
    if (REMOTE_ENGINE_URL) {
        const remoteErrors: string[] = [];
        for (const url of remoteCandidates) {
            try {
                return await executeRemoteSignalEngine(url, payload);
            } catch (err) {
                remoteErrors.push(`${url} -> ${asErrorMessage(err)}`);
            }
        }

        try {
            return await executeLocal(payload);
        } catch (localErr) {
            const details = [...remoteErrors, `local python -> ${asErrorMessage(localErr)}`]
                .map((entry, idx) => `${idx + 1}. ${entry}`)
                .join("\n");
            throw new Error(`Signal engine failed for all execution paths.\n${details}`);
        }
    }

    // Default mode: run local Python directly.
    try {
        return await executeLocal(payload);
    } catch (localErr) {
        const remoteErrors: string[] = [];
        for (const url of remoteCandidates) {
            try {
                return await executeRemoteSignalEngine(url, payload);
            } catch (err) {
                remoteErrors.push(`${url} -> ${asErrorMessage(err)}`);
            }
        }

        const details = [`local python -> ${asErrorMessage(localErr)}`, ...remoteErrors]
            .map((entry, idx) => `${idx + 1}. ${entry}`)
            .join("\n");
        throw new Error(`Signal engine failed for all execution paths.\n${details}`);
    }
}
