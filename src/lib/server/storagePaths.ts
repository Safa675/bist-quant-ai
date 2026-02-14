import { join } from "path";

const explicitDataDir = (process.env.RUN_DATA_DIR || "").trim();

export const DATA_DIR = explicitDataDir
    ? explicitDataDir
    : (process.env.VERCEL
        ? "/tmp/bist-quant-ai-data"
        : join(process.cwd(), "data"));

export const SIGNAL_STORE_PATH = join(DATA_DIR, "signal_store.json");
export const RUN_STORE_PATH = join(DATA_DIR, "run_store.json");
export const ARTIFACTS_DIR = join(DATA_DIR, "artifacts");
