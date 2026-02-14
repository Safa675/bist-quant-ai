import { existsSync } from "fs";
import { mkdir, readFile, rename, stat, writeFile } from "fs/promises";
import { basename, isAbsolute, join, relative, resolve } from "path";
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

function normalizeArtifactId(value: string): string | null {
    const trimmed = (value || "").trim().toLowerCase();
    if (!trimmed) return null;
    if (!/^[a-z0-9_]{1,128}$/.test(trimmed)) return null;
    return trimmed;
}

async function ensureArtifactsDir(): Promise<void> {
    await mkdir(ARTIFACTS_DIR, { recursive: true });
}

function isPathInsideArtifactsDir(candidatePath: string): boolean {
    const artifactsRoot = resolve(ARTIFACTS_DIR);
    const candidate = resolve(candidatePath);
    const rel = relative(artifactsRoot, candidate);
    return rel === "" || (!rel.startsWith("..") && !isAbsolute(rel));
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
    const tempPath = `${filePath}.tmp`;

    await writeFile(tempPath, JSON.stringify(input.payload, null, 2), "utf-8");
    await rename(tempPath, filePath);
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
    if (!path || !isPathInsideArtifactsDir(path) || !existsSync(path)) {
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
    const normalized = normalizeArtifactId(id);
    if (!normalized) {
        return null;
    }
    const candidatePath = join(ARTIFACTS_DIR, `${normalized}.json`);
    return readArtifactByPath(candidatePath);
}

export function artifactFileName(path: string): string {
    return basename(path);
}
