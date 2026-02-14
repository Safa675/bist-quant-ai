import { executeRunNow } from "@/lib/server/runExecution";
import { getRun, updateRun } from "@/lib/server/runStore";

const MAX_CONCURRENT_JOBS = 1;

const pendingQueue: string[] = [];
const pendingSet = new Set<string>();
const activeSet = new Set<string>();

let activeCount = 0;

function isTerminal(status: string | undefined): boolean {
    return status === "succeeded" || status === "failed" || status === "cancelled";
}

function nextTick(fn: () => void): void {
    setTimeout(fn, 0);
}

function isInMemoryQueueDisabled(): boolean {
    const forceDisable = process.env.RUNS_DISABLE_INMEM_QUEUE === "1";
    const serverlessDisable = process.env.VERCEL === "1" && process.env.RUNS_ALLOW_INMEM_QUEUE !== "1";
    return forceDisable || serverlessDisable;
}

async function processSingle(runId: string): Promise<void> {
    try {
        const run = await getRun(runId);
        if (!run || isTerminal(run.status)) {
            return;
        }

        await executeRunNow(run);
    } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        await updateRun({
            id: runId,
            status: "failed",
            error: {
                code: "QUEUE_EXECUTION_ERROR",
                message,
            },
            meta: {
                queue_error: true,
            },
        });
    }
}

function pump(): void {
    while (activeCount < MAX_CONCURRENT_JOBS && pendingQueue.length > 0) {
        const runId = pendingQueue.shift();
        if (!runId) {
            continue;
        }

        pendingSet.delete(runId);
        activeSet.add(runId);
        activeCount += 1;

        void processSingle(runId)
            .finally(() => {
                activeSet.delete(runId);
                activeCount = Math.max(0, activeCount - 1);
                nextTick(pump);
            });
    }
}

export async function enqueueRun(runId: string): Promise<{ position: number; alreadyQueued: boolean }> {
    if (isInMemoryQueueDisabled()) {
        throw new Error("In-memory queue is disabled in this environment. Use blocking execution.");
    }

    const run = await getRun(runId);
    if (!run) {
        throw new Error(`Run not found: ${runId}`);
    }
    if (isTerminal(run.status)) {
        throw new Error(`Run is already terminal (${run.status}): ${runId}`);
    }

    if (pendingSet.has(runId) || activeSet.has(runId)) {
        const idx = pendingQueue.indexOf(runId);
        return {
            position: idx >= 0 ? idx + 1 : 0,
            alreadyQueued: true,
        };
    }

    await updateRun({
        id: runId,
        status: "queued",
        meta: {
            queued_at: new Date().toISOString(),
        },
        error: null,
    });

    pendingQueue.push(runId);
    pendingSet.add(runId);
    nextTick(pump);

    return {
        position: pendingQueue.length,
        alreadyQueued: false,
    };
}

export function getQueueSnapshot(): {
    pending: number;
    active: number;
    max_concurrent: number;
    pending_ids: string[];
    active_ids: string[];
} {
    return {
        pending: pendingQueue.length,
        active: activeCount,
        max_concurrent: MAX_CONCURRENT_JOBS,
        pending_ids: [...pendingQueue],
        active_ids: Array.from(activeSet),
    };
}
