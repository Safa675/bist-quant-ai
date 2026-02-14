import test from "node:test";
import assert from "node:assert/strict";
import type { RunKind } from "../../src/lib/contracts/run";
import { ensureUnitRunDataDir } from "./test-env";

ensureUnitRunDataDir();

test("executeRunNow persists trace_id on dispatch failure", async () => {
    const { createRun, getRun } = await import("../../src/lib/server/runStore");
    const { executeRunNow } = await import("../../src/lib/server/runExecution");

    const run = await createRun({
        kind: "unsupported_kind" as unknown as RunKind,
        request: {},
        status: "queued",
    });

    const outcome = await executeRunNow(run);
    assert.equal(outcome.run.status, "failed");

    const persisted = await getRun(run.id);
    assert.equal(persisted?.status, "failed");
    assert.equal(typeof persisted?.meta?.trace_id, "string");
});
