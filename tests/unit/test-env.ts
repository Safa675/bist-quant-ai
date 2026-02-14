import { mkdirSync } from "fs";
import { join } from "path";

export function ensureUnitRunDataDir(): string {
    const existing = (process.env.RUN_DATA_DIR || "").trim();
    if (existing) {
        mkdirSync(existing, { recursive: true });
        return existing;
    }

    const dir = join(process.cwd(), ".tmp", "unit-run-data");
    process.env.RUN_DATA_DIR = dir;
    mkdirSync(dir, { recursive: true });
    return dir;
}
