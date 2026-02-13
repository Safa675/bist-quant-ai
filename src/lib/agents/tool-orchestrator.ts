/**
 * Tool-Enabled Agent Orchestrator
 *
 * Extends the base orchestrator with function calling capabilities
 * for Borsa MCP integration. Enables LLM agents to:
 * - Screen stocks with fundamental/technical filters
 * - Fetch financial statements and ratios
 * - Run technical scans
 * - Access fund and macro data
 */

import { logError, logInfo } from "./logging";
import {
    executeMCPTool,
    getBorsaMcpToolDefinitions,
    type MCPToolCall,
    type MCPToolResult,
} from "./borsa-mcp-client";

// Re-export types from base orchestrator
export type { AgentContext, AgentMessage } from "./orchestrator";
import type { AgentContext, AgentType } from "./orchestrator";
import { buildContext, getAzureOpenAIConfig, getSystemPrompt, validateQuery } from "./orchestrator";

export type ResearchAgentType = "research";
export type ToolEnabledAgentType = AgentType | ResearchAgentType;

interface ToolCall {
    id: string;
    type: "function";
    function: {
        name: string;
        arguments: string;
    };
}

interface AzureUsage {
    promptTokens: number | null;
    completionTokens: number | null;
    totalTokens: number | null;
}

interface ToolOrchestrationResult {
    response: string;
    toolsUsed: string[];
    toolResults: Record<string, MCPToolResult>;
    usage: AzureUsage;
}

interface ToolOrchestrationOptions {
    requestId?: string;
    maxToolCalls?: number;
    maxRetries?: number;
}

const RESEARCH_SYSTEM_PROMPT = `You are a Research Analyst AI agent for Quant AI Platform.
You focus on deep stock/factor investigation for Borsa Istanbul (BIST).
You can screen names, inspect fundamentals, compare sectors, and summarize findings into actionable views.`;

const MCP_TOOLING_INSTRUCTIONS = `You have access to live Borsa MCP function tools.

Tool usage policy:
1. Use tools whenever the user asks for current prices, screening, rankings, ratios, or news.
2. If a tool fails, briefly report the failure and try an alternative relevant tool or parameter.
3. Never claim you cannot access live market data unless tools actually fail.
4. Keep each tool call lightweight: prefer small batches (e.g., <= 3 symbols per call).
5. Keep responses concise, numerical, and decision-oriented.
6. Prefer BIST-specific analysis and mention symbol/index names explicitly.

Available screening presets: value_stocks, growth_stocks, high_dividend, low_pe, high_momentum, oversold, overbought, breakout, undervalued, quality, small_cap, large_cap
Available scan types: oversold, overbought, macd_bullish, macd_bearish, supertrend_buy, supertrend_sell, golden_cross, death_cross`;

const MAX_TOOL_ITERATIONS = 5;
const DEFAULT_MAX_RETRIES = 2;
const RETRY_DELAY_MS = 1000;
const MAX_CONSECUTIVE_ALL_TOOL_FAILURES = 2;

function sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

function _extractUsage(payload: unknown): AzureUsage {
    const usage = (payload as { usage?: { prompt_tokens?: unknown; completion_tokens?: unknown; total_tokens?: unknown } })?.usage;
    const toNumOrNull = (value: unknown) => {
        const n = Number(value);
        return Number.isFinite(n) ? n : null;
    };
    return {
        promptTokens: toNumOrNull(usage?.prompt_tokens),
        completionTokens: toNumOrNull(usage?.completion_tokens),
        totalTokens: toNumOrNull(usage?.total_tokens),
    };
}

function _extractToolCalls(payload: unknown): ToolCall[] {
    const choices = (payload as { choices?: Array<{ message?: { tool_calls?: ToolCall[] } }> })?.choices;
    return choices?.[0]?.message?.tool_calls || [];
}

function _extractAssistantText(payload: unknown): string {
    const message = (payload as {
        choices?: Array<{
            message?: {
                content?: string | Array<{ text?: string } | string> | null;
                refusal?: string | Array<{ text?: string } | string> | null;
            };
        }>;
    })?.choices?.[0]?.message;

    const content = message?.content;
    if (typeof content === "string") {
        return content.trim();
    }
    if (Array.isArray(content)) {
        const merged = content
            .map((part) => {
                if (typeof part === "string") {
                    return part;
                }
                if (part && typeof part === "object" && typeof part.text === "string") {
                    return part.text;
                }
                return "";
            })
            .join("\n")
            .trim();
        if (merged) {
            return merged;
        }
    }

    const refusal = message?.refusal;
    if (typeof refusal === "string" && refusal.trim()) {
        return refusal.trim();
    }
    if (Array.isArray(refusal)) {
        const merged = refusal
            .map((part) => {
                if (typeof part === "string") {
                    return part;
                }
                if (part && typeof part === "object" && typeof part.text === "string") {
                    return part.text;
                }
                return "";
            })
            .join("\n")
            .trim();
        if (merged) {
            return merged;
        }
    }

    return "";
}

function _extractFinishReason(payload: unknown): string {
    const choices = (payload as { choices?: Array<{ finish_reason?: string }> })?.choices;
    return choices?.[0]?.finish_reason || "";
}

function getToolSystemPrompt(agent: ToolEnabledAgentType): string {
    const basePrompt = agent === "research" ? RESEARCH_SYSTEM_PROMPT : getSystemPrompt(agent);
    return `${basePrompt}\n\n${MCP_TOOLING_INSTRUCTIONS}`;
}

/**
 * Generate an agent response with tool calling support.
 */
export async function generateToolEnabledAgentResponse(
    agent: ToolEnabledAgentType,
    query: string,
    context: AgentContext,
    options: ToolOrchestrationOptions = {}
): Promise<ToolOrchestrationResult> {
    const validation = validateQuery(query);
    if (!validation.valid) {
        throw new Error(validation.error || "Invalid query.");
    }
    const sanitizedQuery = validation.sanitized!;

    const cfg = getAzureOpenAIConfig();
    const url = `${cfg.endpoint}/openai/deployments/${encodeURIComponent(cfg.deployment)}/chat/completions` +
        `?api-version=${encodeURIComponent(cfg.apiVersion)}`;

    const contextStr = buildContext(context);
    const maxRetries = options.maxRetries ?? DEFAULT_MAX_RETRIES;
    const maxIterations = options.maxToolCalls ?? MAX_TOOL_ITERATIONS;
    const toolDefinitions = await getBorsaMcpToolDefinitions(options.requestId);

    const toolsUsed: string[] = [];
    const toolResults: Record<string, MCPToolResult> = {};

    const messages: Array<{
        role: "system" | "user" | "assistant" | "tool";
        content?: string;
        tool_call_id?: string;
        tool_calls?: ToolCall[];
    }> = [
        { role: "system", content: getToolSystemPrompt(agent) },
        {
            role: "user",
            content: `Portfolio Context:\n${contextStr}\n\nUser Question:\n${sanitizedQuery}`,
        },
    ];

    const totalUsage: AzureUsage = {
        promptTokens: null,
        completionTokens: null,
        totalTokens: null,
    };

    let consecutiveAllToolFailures = 0;

    logInfo("tool.agent.start", {
        requestId: options.requestId || null,
        agent,
        queryChars: sanitizedQuery.length,
        toolCount: toolDefinitions.length,
        maxIterations,
    });

    for (let iteration = 0; iteration < maxIterations; iteration++) {
        const startedAt = Date.now();
        let parsed: unknown = null;
        let lastError: Error | null = null;

        for (let attempt = 1; attempt <= maxRetries; attempt++) {
            try {
                const res = await fetch(url, {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                        "api-key": cfg.apiKey,
                    },
                    body: JSON.stringify({
                        messages,
                        tools: toolDefinitions,
                        tool_choice: "auto",
                        max_completion_tokens: 1500,
                    }),
                    cache: "no-store",
                });

                const raw = await res.text();
                try {
                    parsed = JSON.parse(raw);
                } catch {
                    throw new Error(`Invalid JSON response: ${raw.slice(0, 200)}`);
                }

                if (!res.ok) {
                    const errPayload = parsed as { error?: { message?: string } } | null;
                    const upstreamMessage = errPayload?.error?.message || raw.slice(0, 400);
                    throw new Error(`Azure OpenAI error (${res.status}): ${upstreamMessage}`);
                }

                lastError = null;
                break;
            } catch (error) {
                lastError = error instanceof Error ? error : new Error(String(error));
                if (attempt < maxRetries) {
                    await sleep(RETRY_DELAY_MS * attempt);
                }
            }
        }

        if (lastError) {
            throw lastError;
        }

        const latencyMs = Date.now() - startedAt;
        const usage = _extractUsage(parsed);
        const finishReason = _extractFinishReason(parsed);
        const toolCalls = _extractToolCalls(parsed);

        if (usage.promptTokens !== null) {
            totalUsage.promptTokens = (totalUsage.promptTokens || 0) + usage.promptTokens;
        }
        if (usage.completionTokens !== null) {
            totalUsage.completionTokens = (totalUsage.completionTokens || 0) + usage.completionTokens;
        }
        if (usage.totalTokens !== null) {
            totalUsage.totalTokens = (totalUsage.totalTokens || 0) + usage.totalTokens;
        }

        logInfo("tool.agent.iteration", {
            requestId: options.requestId || null,
            agent,
            iteration,
            latencyMs,
            finishReason,
            toolCallCount: toolCalls.length,
            promptTokens: usage.promptTokens,
            completionTokens: usage.completionTokens,
        });

        if (finishReason === "stop" || toolCalls.length === 0) {
            const responseText = _extractAssistantText(parsed);
            const finalResponse = responseText || (
                toolsUsed.length > 0
                    ? "I gathered live data but could not format a final narrative. Please retry with a narrower question."
                    : "I couldn't generate a response. Please try again."
            );

            logInfo("tool.agent.complete", {
                requestId: options.requestId || null,
                agent,
                iterations: iteration + 1,
                toolsUsed,
                responseChars: finalResponse.length,
                totalPromptTokens: totalUsage.promptTokens,
                totalCompletionTokens: totalUsage.completionTokens,
            });

            return {
                response: finalResponse,
                toolsUsed,
                toolResults,
                usage: totalUsage,
            };
        }

        messages.push({
            role: "assistant",
            tool_calls: toolCalls,
        });

        const toolResponses = await Promise.all(toolCalls.map(async (tc) => {
            const toolName = tc.function.name;
            let params: Record<string, unknown> = {};

            try {
                params = JSON.parse(tc.function.arguments || "{}");
            } catch {
                params = {};
            }

            if (!toolsUsed.includes(toolName)) {
                toolsUsed.push(toolName);
            }

            logInfo("tool.agent.tool_call", {
                requestId: options.requestId || null,
                agent,
                tool: toolName,
                params,
                iteration,
            });

            const mcpCall: MCPToolCall = {
                tool: toolName,
                params,
            };

            const result = await executeMCPTool(mcpCall, options.requestId);
            toolResults[`${toolName}_${tc.id}`] = result;

            return {
                id: tc.id,
                toolName,
                result,
            };
        }));

        let allToolCallsFailed = toolResponses.length > 0;
        for (const tr of toolResponses) {
            const content = tr.result.success
                ? JSON.stringify(tr.result.data, null, 2)
                : `Error: ${tr.result.error}`;

            messages.push({
                role: "tool",
                tool_call_id: tr.id,
                content,
            });

            if (tr.result.success) {
                allToolCallsFailed = false;
            }

            logInfo("tool.agent.tool_result", {
                requestId: options.requestId || null,
                agent,
                tool: tr.toolName,
                success: tr.result.success,
                latencyMs: tr.result.latencyMs,
                iteration,
            });
        }

        if (allToolCallsFailed) {
            consecutiveAllToolFailures += 1;
        } else {
            consecutiveAllToolFailures = 0;
        }

        if (consecutiveAllToolFailures >= MAX_CONSECUTIVE_ALL_TOOL_FAILURES) {
            const lastErrors = toolResponses
                .filter((tr) => !tr.result.success)
                .map((tr) => `${tr.toolName}: ${tr.result.error || "unknown error"}`);

            const response = [
                "Live data tools are temporarily failing, so I cannot complete a reliable tool-based answer right now.",
                ...lastErrors.slice(0, 3),
            ].join("\n");

            logError("tool.agent.repeated_tool_failures", {
                requestId: options.requestId || null,
                agent,
                iteration,
                failures: lastErrors,
            });

            return {
                response,
                toolsUsed,
                toolResults,
                usage: totalUsage,
            };
        }
    }

    logError("tool.agent.max_iterations", {
        requestId: options.requestId || null,
        agent,
        maxIterations,
        toolsUsed,
    });

    return {
        response: "I reached the maximum number of tool calls. Please narrow the request and try again.",
        toolsUsed,
        toolResults,
        usage: totalUsage,
    };
}

/**
 * Generate a research response with tool calling
 */
export async function generateResearchResponse(
    query: string,
    context: AgentContext,
    options: ToolOrchestrationOptions = {}
): Promise<ToolOrchestrationResult> {
    return generateToolEnabledAgentResponse("research", query, context, options);
}

/**
 * Simple tool execution without full conversation
 * Useful for one-off data fetches
 */
export async function executeToolDirect(
    toolName: string,
    params: Record<string, unknown>,
    requestId?: string
): Promise<MCPToolResult> {
    const mcpCall: MCPToolCall = {
        tool: toolName,
        params,
    };

    return executeMCPTool(mcpCall, requestId);
}
