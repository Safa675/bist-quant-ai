import { existsSync } from "fs";
import { mkdir, readFile, rename, writeFile } from "fs/promises";
import { dirname, join } from "path";
import { DATA_DIR } from "@/lib/server/storagePaths";
import type { AgentMessage } from "./types";

interface StoredConversation {
    sessionId: string;
    createdAt: string;
    updatedAt: string;
    messages: AgentMessage[];
}

interface ConversationStore {
    conversations: Record<string, StoredConversation>;
}

interface ConversationManagerOptions {
    storePath?: string;
    maxStoredMessageChars?: number;
}

const DEFAULT_STORE_PATH = join(DATA_DIR, "conversation_store.json");
const DEFAULT_SESSION_ID = "default-session";
const DEFAULT_HISTORY_LIMIT = 10;
const DEFAULT_MAX_STORED_MESSAGE_CHARS = 4000;

function emptyStore(): ConversationStore {
    return {
        conversations: {},
    };
}

function toIsoNow(): string {
    return new Date().toISOString();
}

function toTimestampNow(): number {
    return Date.now();
}

function normalizeSessionId(sessionId: string): string {
    const trimmed = (sessionId || "").trim();
    return trimmed ? trimmed.slice(0, 256) : DEFAULT_SESSION_ID;
}

function normalizeHistoryLimit(limit: number): number {
    const numeric = Number(limit);
    if (!Number.isFinite(numeric) || numeric <= 0) {
        return DEFAULT_HISTORY_LIMIT;
    }
    return Math.min(Math.floor(numeric), 200);
}

function normalizeRole(role: string): AgentMessage["role"] {
    return role === "assistant" ? "assistant" : "user";
}

function truncateContent(content: string, maxChars: number): string {
    if (content.length <= maxChars) {
        return content;
    }
    return content.slice(0, maxChars);
}

function parseMessage(raw: unknown): AgentMessage | null {
    if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
        return null;
    }

    const item = raw as {
        role?: unknown;
        content?: unknown;
        timestamp?: unknown;
    };

    const role = item.role === "assistant" ? "assistant" : item.role === "user" ? "user" : null;
    if (!role) {
        return null;
    }

    const content = typeof item.content === "string"
        ? item.content
        : String(item.content ?? "");
    const timestamp = Number(item.timestamp);

    return {
        role,
        content,
        timestamp: Number.isFinite(timestamp) ? timestamp : toTimestampNow(),
    };
}

function parseStore(raw: unknown): ConversationStore {
    if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
        return emptyStore();
    }

    const root = raw as {
        conversations?: unknown;
    };

    if (!root.conversations || typeof root.conversations !== "object" || Array.isArray(root.conversations)) {
        return emptyStore();
    }

    const parsed: ConversationStore = {
        conversations: {},
    };

    for (const [sessionId, value] of Object.entries(root.conversations)) {
        if (!value || typeof value !== "object" || Array.isArray(value)) {
            continue;
        }

        const row = value as {
            sessionId?: unknown;
            createdAt?: unknown;
            updatedAt?: unknown;
            messages?: unknown;
        };

        const normalizedSessionId = normalizeSessionId(typeof row.sessionId === "string" ? row.sessionId : sessionId);
        const messages = Array.isArray(row.messages)
            ? row.messages.map(parseMessage).filter((item): item is AgentMessage => Boolean(item))
            : [];

        parsed.conversations[normalizedSessionId] = {
            sessionId: normalizedSessionId,
            createdAt: typeof row.createdAt === "string" && row.createdAt ? row.createdAt : toIsoNow(),
            updatedAt: typeof row.updatedAt === "string" && row.updatedAt ? row.updatedAt : toIsoNow(),
            messages,
        };
    }

    return parsed;
}

function ensureSession(store: ConversationStore, sessionId: string): StoredConversation {
    const normalizedSessionId = normalizeSessionId(sessionId);
    const existing = store.conversations[normalizedSessionId];
    if (existing) {
        return existing;
    }

    const created: StoredConversation = {
        sessionId: normalizedSessionId,
        createdAt: toIsoNow(),
        updatedAt: toIsoNow(),
        messages: [],
    };
    store.conversations[normalizedSessionId] = created;
    return created;
}

export class ConversationManager {
    private readonly storePath: string;
    private readonly maxStoredMessageChars: number;
    private mutationChain: Promise<void>;

    constructor(options: ConversationManagerOptions = {}) {
        this.storePath = options.storePath || DEFAULT_STORE_PATH;
        this.maxStoredMessageChars = Math.max(
            128,
            Number.isFinite(options.maxStoredMessageChars)
                ? Math.floor(options.maxStoredMessageChars as number)
                : DEFAULT_MAX_STORED_MESSAGE_CHARS
        );
        this.mutationChain = Promise.resolve();
    }

    private async ensureStoreDir(): Promise<void> {
        await mkdir(dirname(this.storePath), { recursive: true });
    }

    private async readStore(): Promise<ConversationStore> {
        await this.ensureStoreDir();
        if (!existsSync(this.storePath)) {
            return emptyStore();
        }

        try {
            const raw = await readFile(this.storePath, "utf-8");
            return parseStore(JSON.parse(raw));
        } catch {
            return emptyStore();
        }
    }

    private async writeStore(store: ConversationStore): Promise<void> {
        await this.ensureStoreDir();
        const tmpPath = `${this.storePath}.${process.pid}.${Date.now()}.tmp`;
        await writeFile(tmpPath, JSON.stringify(store, null, 2), "utf-8");
        await rename(tmpPath, this.storePath);
    }

    private async withExclusiveMutation<T>(mutate: (store: ConversationStore) => Promise<T> | T): Promise<T> {
        const previous = this.mutationChain;
        let release!: () => void;
        this.mutationChain = new Promise<void>((resolve) => {
            release = resolve;
        });

        await previous;
        try {
            const store = await this.readStore();
            const result = await mutate(store);
            await this.writeStore(store);
            return result;
        } finally {
            release();
        }
    }

    async createSession(sessionId: string): Promise<void> {
        const normalizedSessionId = normalizeSessionId(sessionId);
        await this.withExclusiveMutation((store) => {
            ensureSession(store, normalizedSessionId);
        });
    }

    async addMessage(sessionId: string, role: string, content: string): Promise<void> {
        const normalizedSessionId = normalizeSessionId(sessionId);
        const normalizedRole = normalizeRole(role);
        const normalizedContent = truncateContent(String(content || ""), this.maxStoredMessageChars);

        await this.withExclusiveMutation((store) => {
            const session = ensureSession(store, normalizedSessionId);
            session.messages.push({
                role: normalizedRole,
                content: normalizedContent,
                timestamp: toTimestampNow(),
            });
            session.updatedAt = toIsoNow();
        });
    }

    async getConversationHistory(sessionId: string, limit = DEFAULT_HISTORY_LIMIT): Promise<AgentMessage[]> {
        const normalizedSessionId = normalizeSessionId(sessionId);
        const normalizedLimit = normalizeHistoryLimit(limit);
        const store = await this.readStore();
        const session = store.conversations[normalizedSessionId];

        if (!session) {
            return [];
        }

        return session.messages
            .slice(-normalizedLimit)
            .map((message) => ({ ...message }));
    }

    async clearConversation(sessionId: string): Promise<void> {
        const normalizedSessionId = normalizeSessionId(sessionId);
        await this.withExclusiveMutation((store) => {
            const session = ensureSession(store, normalizedSessionId);
            session.messages = [];
            session.updatedAt = toIsoNow();
        });
    }
}
