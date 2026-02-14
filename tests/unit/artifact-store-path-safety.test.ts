import test from "node:test";
import assert from "node:assert/strict";
import { join } from "path";
import { ensureUnitRunDataDir } from "./test-env";

ensureUnitRunDataDir();

test("readArtifactByPath blocks reads outside artifacts directory", async () => {
    const { readArtifactByPath } = await import("../../src/lib/server/artifactStore");

    const outsidePath = join(process.cwd(), "package.json");
    const data = await readArtifactByPath(outsidePath);
    assert.equal(data, null);
});

test("readArtifactById rejects invalid IDs and reads valid IDs", async () => {
    const { saveArtifact, readArtifactById } = await import("../../src/lib/server/artifactStore");

    const saved = await saveArtifact({
        kind: "unit_test",
        runId: "run_1",
        payload: { ok: true },
    });

    const invalid = await readArtifactById("../etc/passwd");
    assert.equal(invalid, null);

    const valid = await readArtifactById(saved.id);
    assert.deepEqual(valid, { ok: true });
});
