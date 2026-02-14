import test from "node:test";
import assert from "node:assert/strict";
import { ensureUnitRunDataDir } from "./test-env";

ensureUnitRunDataDir();

test("createRun preserves all entries under concurrent writes", async () => {
    const { createRun, getRun } = await import("../../src/lib/server/runStore");

    const ids = Array.from({ length: 20 }, (_, i) => `concurrency_case_${Date.now()}_${i}`);

    await Promise.all(ids.map((id) => createRun({
        id,
        kind: "stock_filter",
        request: { idx: id },
        status: "queued",
    })));

    const persisted = await Promise.all(ids.map((id) => getRun(id)));
    const missing = persisted.filter((item) => !item).length;
    assert.equal(missing, 0, `Missing ${missing} runs after concurrent createRun writes.`);
});
