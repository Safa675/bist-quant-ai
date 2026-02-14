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

import { createRequestId, Logger, logError, logInfo } from "./logging";
import { getAgentToolPolicy } from "./policy";
import { createExecutionPlan } from "./planning-agent";
import { ParallelExecutor, type ExecutionStats } from "./parallel-executor";
import { ValidationAgent } from "./validation-agent";
import {
    executeMCPTool,
    getBorsaMcpToolDefinitions,
    type MCPToolCall,
    type MCPToolResult as BorsaMCPToolResult,
} from "./borsa-mcp-client";
import type {
    AgentContext as MultiAgentContext,
    ExecutionPlan,
    MCPToolResult as MultiAgentMCPToolResult,
    TaskNode,
} from "./types";
import { callAzureOpenAI } from "../utils/azure-openai";

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
    toolResults: Record<string, BorsaMCPToolResult>;
    usage: AzureUsage;
}

interface ToolOrchestrationOptions {
    requestId?: string;
    maxToolCalls?: number;
    maxRetries?: number;
    maxConsecutiveToolFailures?: number;
}

export interface OrchestratorConfig {
    maxRetriesPerTask: number;
    maxTotalSteps: number;
    maxConcurrentTasks: number;
}

export interface ComplexQueryResult {
    requestId: string;
    summary: {
        totalTasks: number;
        successfulTasks: number;
        failedTasks: number;
    };
    executionPlan: ExecutionPlan;
    executionStats: ExecutionStats;
    successfulResults: Array<{ taskId: string; data: unknown }>;
    failedResults: Array<{ taskId: string; error?: string }>;
    response: string;
    toolsUsed: string[];
    toolResults: Record<string, MultiAgentMCPToolResult>;
}

function toMultiAgentContext(context: AgentContext | MultiAgentContext, requestId?: string): MultiAgentContext {
    const base = (context || {}) as MultiAgentContext;
    return {
        ...base,
        requestId: requestId || base.requestId,
        regime: base.regime || "Unknown",
        signals: Array.isArray(base.signals) ? base.signals : [],
        holdings: base.holdings && typeof base.holdings === "object" ? base.holdings : {},
    };
}

function normalizeToolResult(result: BorsaMCPToolResult): MultiAgentMCPToolResult {
    return {
        success: result.success,
        data: result.data ?? null,
        error: result.error,
        latencyMs: result.latencyMs,
    };
}

export class ToolOrchestrator {
    private readonly config: OrchestratorConfig;
    private readonly logger: Logger;
    private readonly validationAgent: ValidationAgent;
    private readonly executor: ParallelExecutor;

    constructor(config: Partial<OrchestratorConfig> = {}, logger: Logger = new Logger()) {
        this.config = {
            maxRetriesPerTask: 3,
            maxTotalSteps: 10,
            maxConcurrentTasks: 5,
            ...config,
        };
        this.logger = logger;
        this.validationAgent = new ValidationAgent(logger);
        this.executor = new ParallelExecutor(logger, this.config.maxConcurrentTasks);
    }

    async executeComplexQuery(query: string, context: MultiAgentContext): Promise<ComplexQueryResult> {
        const requestId = context.requestId || createRequestId();
        this.logger.info("agent.complex.start", {
            requestId,
            queryChars: query.length,
        });

        const plan = await createExecutionPlan(query, context);
        this.logger.info("agent.complex.plan_created", {
            requestId,
            taskCount: plan.tasks.length,
            levelCount: plan.executionOrder.length,
        });

        const { results, stats } = await this.executePlanWithValidation(plan, context, requestId);
        const synthesized = await this.synthesizeResults(query, plan, results, stats, context);

        this.logger.info("agent.complex.complete", {
            requestId,
            totalTasks: plan.tasks.length,
            successfulTasks: synthesized.summary.successfulTasks,
            failedTasks: synthesized.summary.failedTasks,
            executionTimeMs: stats.executionTimeMs,
        });

        return synthesized;
    }

    private async executePlanWithValidation(
        plan: ExecutionPlan,
        context: MultiAgentContext,
        requestId: string
    ): Promise<{ results: Map<string, MultiAgentMCPToolResult>; stats: ExecutionStats }> {
        const truncatedPlan = this.truncatePlan(plan, this.config.maxTotalSteps);
        const initial = await this.executor.executeWithDependencies(truncatedPlan, requestId);
        const results = new Map(initial.results);
        const taskMap = new Map(truncatedPlan.tasks.map((task) => [task.id, task]));

        for (const level of truncatedPlan.executionOrder) {
            for (const taskId of level) {
                const task = taskMap.get(taskId);
                if (!task) {
                    continue;
                }

                const dependenciesSatisfied = task.dependencies.every((depId) => {
                    const depResult = results.get(depId);
                    return Boolean(depResult?.success);
                });

                if (!dependenciesSatisfied) {
                    results.set(taskId, {
                        success: false,
                        data: null,
                        error: "Dependencies not satisfied",
                    });
                    continue;
                }

                let currentResult = results.get(taskId) || {
                    success: false,
                    data: null,
                    error: "No result returned from executor",
                };

                let validation = await this.validationAgent.validateToolResult(task, currentResult, context);
                let attempts = 1;

                while (!validation.isValid && attempts < this.config.maxRetriesPerTask) {
                    if (validation.suggestedAction?.type === "skip") {
                        break;
                    }

                    const retryParams = validation.suggestedAction?.params && typeof validation.suggestedAction.params === "object"
                        ? validation.suggestedAction.params
                        : this.enrichTaskParams(task, results);
                    attempts += 1;

                    this.logger.warn("agent.complex.retry", {
                        requestId,
                        taskId: task.id,
                        attempt: attempts,
                        issues: validation.issues,
                    });

                    await sleep(Math.min(1500, RETRY_DELAY_MS * attempts));
                    currentResult = await this.executeSingleTask(task, retryParams, requestId, attempts);
                    validation = await this.validationAgent.validateToolResult(task, currentResult, context);
                }

                results.set(taskId, currentResult);
            }
        }

        const completedTasks = Array.from(results.values()).filter((result) => result.success).length;
        const failedTasks = truncatedPlan.tasks.length - completedTasks;
        const stats: ExecutionStats = {
            ...initial.stats,
            totalTasks: truncatedPlan.tasks.length,
            completedTasks,
            failedTasks,
            parallelEfficiency: truncatedPlan.tasks.length > 0
                ? completedTasks / truncatedPlan.tasks.length
                : 0,
        };

        return { results, stats };
    }

    private truncatePlan(plan: ExecutionPlan, maxTasks: number): ExecutionPlan {
        if (plan.tasks.length <= maxTasks) {
            return plan;
        }

        const allowed = new Set(plan.tasks.slice(0, maxTasks).map((task) => task.id));
        const tasks = plan.tasks.filter((task) => allowed.has(task.id)).map((task) => ({
            ...task,
            dependencies: task.dependencies.filter((dep) => allowed.has(dep)),
        }));
        const executionOrder = plan.executionOrder
            .map((level) => level.filter((taskId) => allowed.has(taskId)))
            .filter((level) => level.length > 0);

        this.logger.warn("agent.complex.plan_truncated", {
            originalTaskCount: plan.tasks.length,
            truncatedTaskCount: tasks.length,
            maxTasks,
        });

        return { tasks, executionOrder };
    }

    private enrichTaskParams(
        task: TaskNode,
        results: Map<string, MultiAgentMCPToolResult>
    ): Record<string, unknown> {
        const enriched: Record<string, unknown> = { ...task.params };
        for (const depId of task.dependencies) {
            const depResult = results.get(depId);
            if (depResult?.success && depResult.data !== undefined) {
                enriched[`_dep_${depId.replace(/[^a-zA-Z0-9]/g, "_")}`] = depResult.data;
            }
        }
        return enriched;
    }

    private async executeSingleTask(
        task: TaskNode,
        params: Record<string, unknown>,
        requestId: string,
        attempt: number
    ): Promise<MultiAgentMCPToolResult> {
        this.logger.debug("agent.complex.task_execute", {
            requestId,
            taskId: task.id,
            tool: task.tool,
            attempt,
        });
        const result = await executeMCPTool(
            { tool: task.tool, params },
            `${requestId}-${task.id}-attempt-${attempt}`
        );
        return normalizeToolResult(result);
    }

    private async synthesizeResults(
        query: string,
        plan: ExecutionPlan,
        results: Map<string, MultiAgentMCPToolResult>,
        stats: ExecutionStats,
        context: MultiAgentContext
    ): Promise<ComplexQueryResult> {
        const successfulResults = Array.from(results.entries())
            .filter(([, result]) => result.success)
            .map(([taskId, result]) => ({ taskId, data: result.data }));
        const failedResults = Array.from(results.entries())
            .filter(([, result]) => !result.success)
            .map(([taskId, result]) => ({ taskId, error: result.error }));
        const toolsUsed = Array.from(
            new Set(
                plan.tasks
                    .filter((task) => results.get(task.id)?.success)
                    .map((task) => task.tool)
            )
        );
        const toolResults: Record<string, MultiAgentMCPToolResult> = {};
        for (const [taskId, result] of results.entries()) {
            toolResults[taskId] = result;
        }

        const requestId = context.requestId || createRequestId();
        const summary = {
            totalTasks: plan.tasks.length,
            successfulTasks: successfulResults.length,
            failedTasks: failedResults.length,
        };

        let response = "";
        try {
            const answerPrompt = `You are an answer synthesis agent.
Given the query, execution summary, and tool outputs, write a concise actionable response.

Query:
${query}

Summary:
${JSON.stringify(summary, null, 2)}

Successful task outputs:
${JSON.stringify(successfulResults.slice(0, 8), null, 2)}

Failed tasks:
${JSON.stringify(failedResults.slice(0, 8), null, 2)}

Return plain text only.`;
            response = await callAzureOpenAI(answerPrompt, {
                temperature: 0.2,
                max_tokens: 900,
                requestId,
            });
        } catch {
            response = [
                `Completed ${summary.successfulTasks}/${summary.totalTasks} tasks.`,
                summary.failedTasks > 0 ? `Failed tasks: ${summary.failedTasks}.` : "No task failures.",
                "See tool outputs for details.",
            ].join(" ");
        }

        return {
            requestId,
            summary,
            executionPlan: plan,
            executionStats: stats,
            successfulResults,
            failedResults,
            response,
            toolsUsed,
            toolResults,
        };
    }
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

const RETRY_DELAY_MS = 1000;

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
    const policy = getAgentToolPolicy(agent);
    const dynamicToolingInstructions = MCP_TOOLING_INSTRUCTIONS.replace("<= 3", `<= ${policy.maxSymbolsPerToolCall}`);
    const basePrompt = agent === "research" ? RESEARCH_SYSTEM_PROMPT : getSystemPrompt(agent);
    return `${basePrompt}\n\n${dynamicToolingInstructions}`;
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

    const phase2Enabled = !["0", "false", "off", "no"].includes(
        (process.env.AGENT_PHASE2_ENABLED || "1").trim().toLowerCase()
    );

    if (phase2Enabled) {
        const requestId = options.requestId || createRequestId();
        try {
            const policy = getAgentToolPolicy(agent);
            const orchestrator = new ToolOrchestrator(
                {
                    maxRetriesPerTask: options.maxRetries ?? policy.maxRetries,
                    maxTotalSteps: options.maxToolCalls ?? policy.maxToolCalls,
                    maxConcurrentTasks: Math.max(1, policy.maxSymbolsPerToolCall),
                },
                new Logger({ requestId, agent, mode: "phase2" })
            );
            const complexContext = toMultiAgentContext(context, requestId);
            const complexResult = await orchestrator.executeComplexQuery(sanitizedQuery, complexContext);

            const toolResults: Record<string, BorsaMCPToolResult> = {};
            for (const [taskId, result] of Object.entries(complexResult.toolResults)) {
                toolResults[taskId] = {
                    success: result.success,
                    data: result.data,
                    error: result.error,
                    latencyMs: result.latencyMs ?? 0,
                };
            }

            return {
                response: complexResult.response,
                toolsUsed: complexResult.toolsUsed,
                toolResults,
                usage: {
                    promptTokens: null,
                    completionTokens: null,
                    totalTokens: null,
                },
            };
        } catch (error) {
            logError("tool.agent.phase2.fallback_legacy", {
                requestId,
                agent,
                error: error instanceof Error ? error.message : String(error),
            });
        }
    }

    const cfg = getAzureOpenAIConfig();
    const url = `${cfg.endpoint}/openai/deployments/${encodeURIComponent(cfg.deployment)}/chat/completions` +
        `?api-version=${encodeURIComponent(cfg.apiVersion)}`;

    const contextStr = buildContext(context);
    const policy = getAgentToolPolicy(agent);
    const maxRetries = options.maxRetries ?? policy.maxRetries;
    const maxIterations = options.maxToolCalls ?? policy.maxToolCalls;
    const maxConsecutiveAllToolFailures = options.maxConsecutiveToolFailures ?? policy.maxConsecutiveToolFailures;
    const toolDefinitions = await getBorsaMcpToolDefinitions(options.requestId);

    const toolsUsed: string[] = [];
    const toolResults: Record<string, BorsaMCPToolResult> = {};

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
        maxRetries,
        maxConsecutiveAllToolFailures,
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

        if (consecutiveAllToolFailures >= maxConsecutiveAllToolFailures) {
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
        response: `I reached the maximum number of tool calls (${maxIterations}). Please narrow the request and try again.`,
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
): Promise<BorsaMCPToolResult> {
    const mcpCall: MCPToolCall = {
        tool: toolName,
        params,
    };

    return executeMCPTool(mcpCall, requestId);
}
