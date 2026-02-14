import test from "node:test";
import assert from "node:assert/strict";
import { NextRequest } from "next/server";
import { ensureUnitRunDataDir } from "./test-env";

ensureUnitRunDataDir();

test("GET /api/runs/[runId]?include_artifact=1 ignores mutable meta.artifact_path", async () => {
    const { createRun } = await import("../../src/lib/server/runStore");
    const { GET } = await import("../../src/app/api/runs/[runId]/route");

    const run = await createRun({
        kind: "stock_filter",
        request: {},
        status: "succeeded",
        meta: {
            artifact_path: "/etc/hosts",
        },
    });

    const request = new NextRequest(`http://localhost/api/runs/${run.id}?include_artifact=1`);
    const response = await GET(request, { params: Promise.resolve({ runId: run.id }) });
    assert.equal(response.status, 200);

    const payload = await response.json() as { artifact?: unknown };
    assert.equal(payload.artifact ?? null, null);
});
