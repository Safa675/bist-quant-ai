import { randomUUID } from "crypto";
import { emitStructuredLog } from "@/lib/server/telemetry";

type LogLevel = "debug" | "info" | "warn" | "error";

type LogMeta = Record<string, unknown>;

function emitLog(level: LogLevel, event: string, meta: LogMeta = {}) {
    const telemetryLevel = level === "error" ? "error" : "info";
    emitStructuredLog(telemetryLevel, event, {
        log_level: level,
        ...meta,
    });
}

export function createRequestId(): string {
    return randomUUID();
}

export function logInfo(event: string, meta: LogMeta = {}) {
    emitLog("info", event, meta);
}

export function logDebug(event: string, meta: LogMeta = {}) {
    emitLog("debug", event, meta);
}

export function logWarn(event: string, meta: LogMeta = {}) {
    emitLog("warn", event, meta);
}

export function logError(event: string, meta: LogMeta = {}) {
    emitLog("error", event, meta);
}

export function safeErrorMessage(error: unknown): string {
    if (error instanceof Error && error.message) {
        return error.message;
    }
    if (typeof error === "string" && error.trim()) {
        return error.trim();
    }
    return "Unknown error";
}

export class Logger {
    private readonly baseMeta: LogMeta;

    constructor(baseMeta: LogMeta = {}) {
        this.baseMeta = baseMeta;
    }

    debug(message: string, meta: LogMeta = {}) {
        emitLog("debug", message, { ...this.baseMeta, ...meta });
    }

    info(message: string, meta: LogMeta = {}) {
        emitLog("info", message, { ...this.baseMeta, ...meta });
    }

    warn(message: string, meta: LogMeta = {}) {
        emitLog("warn", message, { ...this.baseMeta, ...meta });
    }

    error(message: string, meta: LogMeta = {}) {
        emitLog("error", message, { ...this.baseMeta, ...meta });
    }
}
