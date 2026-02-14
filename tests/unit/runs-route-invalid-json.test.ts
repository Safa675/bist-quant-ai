import test from "node:test";
import assert from "node:assert/strict";
import { NextRequest } from "next/server";
import { ensureUnitRunDataDir } from "./test-env";

ensureUnitRunDataDir();

test("POST /api/runs returns 400 for malformed JSON", async () => {
    const { POST } = await import("../../src/app/api/runs/route");

    const request = new NextRequest("http://localhost/api/runs", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: "{",
    });

    const response = await POST(request);
    assert.equal(response.status, 400);

    const payload = await response.json() as { error?: string };
    assert.match(String(payload.error || ""), /json object/i);
});
