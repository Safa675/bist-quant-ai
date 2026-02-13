import { existsSync } from "fs";
import { mkdir, readFile, stat, writeFile } from "fs/promises";
import { basename, join } from "path";
import { ARTIFACTS_DIR } from "@/lib/server/storagePaths";

export interface StoredArtifact {
    id: string;
    kind: string;
    path: string;
    created_at: string;
    size_bytes: number;
}

function nowIso(): string {
    return new Date().toISOString();
}

function sanitize(value: string): string {
    return value
        .trim()
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "_")
        .replace(/^_+|_+$/g, "")
        .slice(0, 64) || "artifact";
}

async function ensureArtifactsDir(): Promise<void> {
    await mkdir(ARTIFACTS_DIR, { recursive: true });
}

export async function saveArtifact(input: {
    kind: string;
    payload: unknown;
    runId?: string;
}): Promise<StoredArtifact> {
    await ensureArtifactsDir();

    const kind = sanitize(input.kind || "artifact");
    const runPart = input.runId ? sanitize(input.runId) : "run";
    const id = `${kind}_${runPart}_${Date.now()}`;
    const fileName = `${id}.json`;
    const filePath = join(ARTIFACTS_DIR, fileName);

    await writeFile(filePath, JSON.stringify(input.payload, null, 2), "utf-8");
    const info = await stat(filePath);

    return {
        id,
        kind,
        path: filePath,
        created_at: nowIso(),
        size_bytes: info.size,
    };
}

export async function readArtifactByPath(path: string): Promise<unknown | null> {
    if (!path || !existsSync(path)) {
        return null;
    }

    const raw = await readFile(path, "utf-8");
    try {
        return JSON.parse(raw);
    } catch {
        return raw;
    }
}

export async function readArtifactById(id: string): Promise<unknown | null> {
    const safe = sanitize(id);
    const candidatePath = join(ARTIFACTS_DIR, `${safe}.json`);
    return readArtifactByPath(candidatePath);
}

export function artifactFileName(path: string): string {
    return basename(path);
}
