export type AgentType = "portfolio" | "risk" | "analyst" | "research";

export interface AgentMessage {
    role: "user" | "assistant";
    content: string;
    timestamp: number;
}

export interface AgentContext {
    userId?: string;
    sessionId?: string;
    requestId?: string;
    conversationHistory?: AgentMessage[];
    preferences?: {
        riskTolerance?: "conservative" | "moderate" | "aggressive";
        investmentHorizon?: "short" | "medium" | "long";
    };
    regime?: string;
    signals?: Array<{
        name: string;
        cagr: number;
        sharpe: number;
        max_dd: number;
        ytd: number;
    }>;
    holdings?: Record<string, string[]>;
}

export interface MCPToolResult {
    success: boolean;
    data: unknown;
    error?: string;
    latencyMs?: number;
}

export interface ToolCall {
    tool: string;
    params: Record<string, unknown>;
}

export interface TaskNode {
    id: string;
    description: string;
    tool: string;
    params: Record<string, unknown>;
    dependencies: string[];
    status: "pending" | "running" | "completed" | "failed";
    result?: unknown;
    priority: number;
}

export interface ExecutionPlan {
    tasks: TaskNode[];
    executionOrder: string[][];
}
