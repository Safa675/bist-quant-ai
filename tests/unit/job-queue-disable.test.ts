import test from "node:test";
import assert from "node:assert/strict";
import { ensureUnitRunDataDir } from "./test-env";

ensureUnitRunDataDir();

test("enqueueRun rejects when in-memory queue is disabled", async () => {
    const { createRun } = await import("../../src/lib/server/runStore");
    const { enqueueRun } = await import("../../src/lib/server/jobQueue");

    const run = await createRun({
        kind: "stock_filter",
        request: { sample: true },
        status: "queued",
    });

    const previous = process.env.RUNS_DISABLE_INMEM_QUEUE;
    process.env.RUNS_DISABLE_INMEM_QUEUE = "1";
    try {
        await assert.rejects(
            () => enqueueRun(run.id),
            /disabled/i,
        );
    } finally {
        if (typeof previous === "string") {
            process.env.RUNS_DISABLE_INMEM_QUEUE = previous;
        } else {
            delete process.env.RUNS_DISABLE_INMEM_QUEUE;
        }
    }
});
