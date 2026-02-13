import { join } from "path";

export const DATA_DIR = process.env.VERCEL
    ? "/tmp/bist-quant-ai-data"
    : join(process.cwd(), "data");

export const SIGNAL_STORE_PATH = join(DATA_DIR, "signal_store.json");
export const RUN_STORE_PATH = join(DATA_DIR, "run_store.json");
export const ARTIFACTS_DIR = join(DATA_DIR, "artifacts");
