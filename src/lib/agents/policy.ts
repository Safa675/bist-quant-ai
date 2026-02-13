export type AgentPolicyType = "portfolio" | "risk" | "analyst" | "research";

export interface AgentToolPolicy {
    maxToolCalls: number;
    maxRetries: number;
    maxConsecutiveToolFailures: number;
    maxSymbolsPerToolCall: number;
}

const DEFAULT_POLICY: AgentToolPolicy = {
    maxToolCalls: 10,
    maxRetries: 2,
    maxConsecutiveToolFailures: 2,
    maxSymbolsPerToolCall: 6,
};

const AGENT_POLICIES: Record<AgentPolicyType, AgentToolPolicy> = {
    portfolio: {
        ...DEFAULT_POLICY,
        maxToolCalls: 10,
        maxSymbolsPerToolCall: 5,
    },
    risk: {
        ...DEFAULT_POLICY,
        maxToolCalls: 10,
        maxSymbolsPerToolCall: 5,
    },
    analyst: {
        ...DEFAULT_POLICY,
        maxToolCalls: 12,
        maxSymbolsPerToolCall: 6,
    },
    research: {
        ...DEFAULT_POLICY,
        maxToolCalls: 18,
        maxSymbolsPerToolCall: 8,
    },
};

function clampInt(value: number, minimum: number, maximum: number): number {
    if (!Number.isFinite(value)) return minimum;
    return Math.max(minimum, Math.min(maximum, Math.trunc(value)));
}

function parseEnvInt(name: string): number | null {
    const raw = process.env[name];
    if (!raw) return null;
    const parsed = Number.parseInt(raw, 10);
    if (!Number.isFinite(parsed)) return null;
    return parsed;
}

function envOverride(name: string, fallback: number, minimum: number, maximum: number): number {
    const parsed = parseEnvInt(name);
    if (parsed === null) return fallback;
    return clampInt(parsed, minimum, maximum);
}

export function getAgentToolPolicy(agent: AgentPolicyType): AgentToolPolicy {
    const base = AGENT_POLICIES[agent] || DEFAULT_POLICY;

    const globalMaxToolCalls = parseEnvInt("AGENT_MAX_TOOL_CALLS");
    const agentMaxToolCalls = parseEnvInt(`AGENT_MAX_TOOL_CALLS_${agent.toUpperCase()}`);
    const resolvedMaxToolCalls = clampInt(
        agentMaxToolCalls ?? globalMaxToolCalls ?? base.maxToolCalls,
        1,
        100,
    );

    return {
        maxToolCalls: resolvedMaxToolCalls,
        maxRetries: envOverride("AGENT_TOOL_MAX_RETRIES", base.maxRetries, 1, 8),
        maxConsecutiveToolFailures: envOverride(
            "AGENT_TOOL_MAX_CONSECUTIVE_FAILURES",
            base.maxConsecutiveToolFailures,
            1,
            10,
        ),
        maxSymbolsPerToolCall: envOverride(
            "AGENT_TOOL_MAX_SYMBOLS_PER_CALL",
            base.maxSymbolsPerToolCall,
            1,
            30,
        ),
    };
}
