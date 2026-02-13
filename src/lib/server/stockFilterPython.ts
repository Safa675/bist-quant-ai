import { spawn } from "child_process";
import path from "path";

const PYTHON_SCRIPT = path.resolve(process.cwd(), "dashboard", "stock_filter_api.py");
const REMOTE_ENGINE_URL = (process.env.STOCK_FILTER_ENGINE_URL || "").trim();

const LOCAL_STOCK_FILTER_API_PATHS = [
    "/py-api/api/stock_filter",
    "/api/stock_filter",
    "/api/index.py/api/stock_filter",
];

export interface StockFilterPayload {
    _mode?: "meta" | "run";
    template?: string;
    sector?: string;
    index?: string;
    recommendation?: string;
    sort_by?: string;
    sort_desc?: boolean;
    limit?: number;
    columns?: string[];
    filters?: Record<string, { min?: number | null; max?: number | null }>;
    percentile_filters?: Record<string, { min_pct?: number | null; max_pct?: number | null }>;
}

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

async function executeRemote(url: string, payload: StockFilterPayload): Promise<Record<string, unknown>> {
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

    return result;
}

async function executeLocal(payload: StockFilterPayload): Promise<Record<string, unknown>> {
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
                resolve(parseJsonFromStdout(stdout));
            } catch (err) {
                const msg = err instanceof Error ? err.message : String(err);
                reject(new Error(`${msg}\nStderr: ${stderr || "(none)"}`));
            }
        });

        child.stdin.write(JSON.stringify(payload));
        child.stdin.end();
    });
}

export async function executeStockFilterPython(payload: StockFilterPayload): Promise<Record<string, unknown>> {
    const remoteCandidates = buildRemoteCandidates();

    // Always prefer local python for consistency with local codebase.
    try {
        return await executeLocal(payload);
    } catch (localErr) {
        const remoteErrors: string[] = [];
        for (const url of remoteCandidates) {
            try {
                return await executeRemote(url, payload);
            } catch (err) {
                remoteErrors.push(`${url} -> ${asErrorMessage(err)}`);
            }
        }

        const details = [`local python -> ${asErrorMessage(localErr)}`, ...remoteErrors]
            .map((entry, idx) => `${idx + 1}. ${entry}`)
            .join("\n");
        throw new Error(`Stock filter engine failed for all execution paths.\n${details}`);
    }
}
