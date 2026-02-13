import { randomUUID } from "crypto";

export type TelemetryLevel = "info" | "error";
export type TelemetryMeta = Record<string, unknown>;

function normalizeTraceCandidate(value: unknown): string | null {
    if (typeof value !== "string") {
        return null;
    }

    const trimmed = value.trim();
    if (!trimmed) {
        return null;
    }

    return trimmed.slice(0, 128);
}

export function createTraceId(): string {
    return randomUUID();
}

export function resolveTraceId(...candidates: unknown[]): string {
    for (const candidate of candidates) {
        const normalized = normalizeTraceCandidate(candidate);
        if (normalized) {
            return normalized;
        }
    }
    return createTraceId();
}

export function traceIdFromHeaders(headers: Headers | null | undefined): string | null {
    if (!headers) {
        return null;
    }

    return normalizeTraceCandidate(
        headers.get("x-trace-id"),
    ) || normalizeTraceCandidate(
        headers.get("x-request-id"),
    ) || normalizeTraceCandidate(
        headers.get("x-correlation-id"),
    );
}

function toErrorMeta(error: unknown): { message: string; name?: string } {
    if (error instanceof Error) {
        return {
            message: error.message || "Unknown error",
            name: error.name || undefined,
        };
    }

    if (typeof error === "string" && error.trim()) {
        return { message: error.trim() };
    }

    return { message: "Unknown error" };
}

export function emitStructuredLog(level: TelemetryLevel, event: string, meta: TelemetryMeta = {}): void {
    const payload = {
        ts: new Date().toISOString(),
        level,
        event,
        ...meta,
    };

    const line = JSON.stringify(payload);
    if (level === "error") {
        console.error(line);
    } else {
        console.info(line);
    }
}

export function telemetryInfo(event: string, meta: TelemetryMeta = {}): void {
    emitStructuredLog("info", event, meta);
}

export function telemetryError(event: string, error: unknown, meta: TelemetryMeta = {}): void {
    const err = toErrorMeta(error);
    emitStructuredLog("error", event, {
        ...meta,
        error_message: err.message,
        error_name: err.name,
    });
}

export function buildTraceHeaders(
    traceId: string,
    headers?: Record<string, string>,
): Record<string, string> {
    return {
        ...(headers || {}),
        "X-Trace-Id": traceId,
    };
}

export function requestTraceId(request: { headers: Headers }, fallback?: string): string {
    return resolveTraceId(traceIdFromHeaders(request.headers), fallback);
}

export function elapsedMs(startedAtMs: number): number {
    return Math.max(0, Date.now() - startedAtMs);
}
