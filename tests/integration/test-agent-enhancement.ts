import assert from "node:assert/strict";
import test from "node:test";
import { join } from "path";

import { ConversationManager } from "../../src/lib/agents/conversation-manager";
import { MainOrchestrator } from "../../src/lib/agents/orchestrator";
import { topologicalSort } from "../../src/lib/agents/planning-agent";
import { Logger } from "../../src/lib/agents/logging";
import { ToolOrchestrator } from "../../src/lib/agents/tool-orchestrator";
import { ValidationAgent } from "../../src/lib/agents/validation-agent";
import type { ExecutionPlan, TaskNode } from "../../src/lib/agents/types";

function makeStorePath(tag: string): string {
    return join(
        process.cwd(),
        ".tmp",
        "agent-enhancement",
        `${tag}_${Date.now()}_${Math.random().toString(36).slice(2, 10)}.json`
    );
}

function mockComplexResult(response: string) {
    return {
        requestId: "test-request",
        summary: {
            totalTasks: 0,
            successfulTasks: 0,
            failedTasks: 0,
        },
        executionPlan: {
            tasks: [],
            executionOrder: [],
        },
        executionStats: {
            startTime: Date.now(),
            endTime: Date.now(),
            totalTasks: 0,
            completedTasks: 0,
            failedTasks: 0,
            executionTimeMs: 0,
            parallelEfficiency: 1,
        },
        successfulResults: [],
        failedResults: [],
        response,
        toolsUsed: [],
        toolResults: {},
    };
}

function patchExecuteComplexQuery(
    handler: (query: string, context: Record<string, unknown>) => Promise<unknown>
): () => void {
    const original = ToolOrchestrator.prototype.executeComplexQuery;
    (ToolOrchestrator.prototype as unknown as {
        executeComplexQuery: (
            query: string,
            context: Record<string, unknown>
        ) => Promise<unknown>;
    }).executeComplexQuery = async function (query: string, context: Record<string, unknown>) {
        return handler(query, context);
    };

    return () => {
        ToolOrchestrator.prototype.executeComplexQuery = original;
    };
}

test("phase2 planning: topological sort respects dependencies and priorities", () => {
    const tasks: TaskNode[] = [
        {
            id: "t1",
            description: "search symbol",
            tool: "search_symbol",
            params: { query: "bank" },
            dependencies: [],
            status: "pending",
            priority: 2,
        },
        {
            id: "t2",
            description: "screen",
            tool: "screen_securities",
            params: { preset: "high_dividend" },
            dependencies: ["t1"],
            status: "pending",
            priority: 5,
        },
        {
            id: "t3",
            description: "technical",
            tool: "get_technical_analysis",
            params: { symbol: "THYAO" },
            dependencies: ["t1"],
            status: "pending",
            priority: 4,
        },
    ];

    const levels = topologicalSort(tasks);
    assert.equal(levels.length, 2);
    assert.deepEqual(levels[0], ["t1"]);
    assert.deepEqual(levels[1], ["t2", "t3"]);
});

test("phase2 validation: screener empty result is invalid", async () => {
    const agent = new ValidationAgent(new Logger({ test: "validation" }));
    const task: TaskNode = {
        id: "screen_1",
        description: "screen names",
        tool: "screen_securities",
        params: { preset: "value_stocks" },
        dependencies: [],
        status: "pending",
        priority: 3,
    };

    const result = await agent.validateToolResult(
        task,
        { success: true, data: [] },
        { requestId: "test-validation" }
    );

    assert.equal(result.isValid, false);
    assert.ok(result.issues.some((issue) => issue.toLowerCase().includes("empty") || issue.toLowerCase().includes("zero")));
});

test("phase2 orchestration: retries invalid task results", async () => {
    const orchestrator = new ToolOrchestrator(
        { maxRetriesPerTask: 2, maxTotalSteps: 5, maxConcurrentTasks: 2 },
        new Logger({ test: "retry" })
    ) as unknown as {
        executePlanWithValidation: (
            plan: ExecutionPlan,
            context: { requestId?: string },
            requestId: string
        ) => Promise<{ results: Map<string, { success: boolean; data: unknown; error?: string }> }>;
        executor: {
            executeWithDependencies: (
                _plan: ExecutionPlan,
                _requestId: string
            ) => Promise<{
                results: Map<string, { success: boolean; data: unknown; error?: string }>;
                stats: {
                    startTime: number;
                    endTime: number;
                    totalTasks: number;
                    completedTasks: number;
                    failedTasks: number;
                    executionTimeMs: number;
                    parallelEfficiency: number;
                };
            }>;
        };
        validationAgent: {
            validateToolResult: (...args: unknown[]) => Promise<{
                isValid: boolean;
                issues: string[];
                confidence: number;
                suggestedAction?: {
                    type: "retry" | "skip" | "modify";
                    params?: Record<string, unknown>;
                };
                metadata: { validationTimestamp: number; validatorVersion: string };
            }>;
        };
        executeSingleTask: (...args: unknown[]) => Promise<{ success: boolean; data: unknown; error?: string }>;
    };

    let validationCallCount = 0;
    let retryCallCount = 0;

    orchestrator.executor = {
        executeWithDependencies: async () => ({
            results: new Map([[
                "task_1",
                { success: false, data: null, error: "initial failure" },
            ]]),
            stats: {
                startTime: Date.now(),
                endTime: Date.now(),
                totalTasks: 1,
                completedTasks: 0,
                failedTasks: 1,
                executionTimeMs: 1,
                parallelEfficiency: 0,
            },
        }),
    };

    orchestrator.validationAgent = {
        validateToolResult: async () => {
            validationCallCount += 1;
            if (validationCallCount === 1) {
                return {
                    isValid: false,
                    issues: ["retry please"],
                    confidence: 0.2,
                    suggestedAction: {
                        type: "retry",
                        params: { symbol: "THYAO" },
                    },
                    metadata: { validationTimestamp: Date.now(), validatorVersion: "1.0" },
                };
            }
            return {
                isValid: true,
                issues: [],
                confidence: 1,
                metadata: { validationTimestamp: Date.now(), validatorVersion: "1.0" },
            };
        },
    };

    orchestrator.executeSingleTask = async () => {
        retryCallCount += 1;
        return { success: true, data: { ok: true } };
    };

    const plan: ExecutionPlan = {
        tasks: [
            {
                id: "task_1",
                description: "task",
                tool: "get_quick_info",
                params: { symbol: "THYAO" },
                dependencies: [],
                status: "pending",
                priority: 3,
            },
        ],
        executionOrder: [["task_1"]],
    };

    const out = await orchestrator.executePlanWithValidation(plan, { requestId: "test" }, "test");
    assert.equal(retryCallCount, 1);
    assert.equal(out.results.get("task_1")?.success, true);
});

test("phase2 orchestrator: main orchestrator is constructible", () => {
    const orchestrator = new MainOrchestrator();
    assert.ok(orchestrator);
});

test("phase5 orchestrator: persists conversation history across calls", async () => {
    const conversationManager = new ConversationManager({
        storePath: makeStorePath("phase5_persist"),
    });
    const orchestrator = new MainOrchestrator(conversationManager);

    const historyLengths: number[] = [];
    const restore = patchExecuteComplexQuery(async (_query, context) => {
        const history = Array.isArray(context.conversationHistory)
            ? context.conversationHistory
            : [];
        historyLengths.push(history.length);
        return mockComplexResult(`assistant-reply-${historyLengths.length}`);
    });

    try {
        await orchestrator.processQuery("First user message", {
            sessionId: "session-persist",
        });
        await orchestrator.processQuery("Second user message", {
            sessionId: "session-persist",
        });
    } finally {
        restore();
    }

    assert.deepEqual(historyLengths, [1, 3]);

    const persisted = await conversationManager.getConversationHistory("session-persist", 10);
    assert.equal(persisted.length, 4);
    assert.equal(persisted[0].role, "user");
    assert.equal(persisted[1].role, "assistant");
    assert.equal(persisted[2].role, "user");
    assert.equal(persisted[3].role, "assistant");
});

test("phase5 orchestrator: history is limited and token-optimized", async () => {
    const conversationManager = new ConversationManager({
        storePath: makeStorePath("phase5_limits"),
    });
    const sessionId = "session-limits";

    const seedMessages = [
        ["user", "u1"],
        ["assistant", "a1"],
        ["user", "u2"],
        ["assistant", "a2"],
        ["user", "u3"],
        ["assistant", "a3"],
        ["user", "u4"],
        ["assistant", "a4"],
    ] as const;

    for (const [role, content] of seedMessages) {
        await conversationManager.addMessage(sessionId, role, content);
    }

    const orchestrator = new MainOrchestrator(conversationManager);
    const capturedHistory: Array<{ role: string; content: string }> = [];
    const restore = patchExecuteComplexQuery(async (_query, context) => {
        if (Array.isArray(context.conversationHistory)) {
            for (const item of context.conversationHistory) {
                if (!item || typeof item !== "object") continue;
                capturedHistory.push({
                    role: String((item as { role?: unknown }).role || ""),
                    content: String((item as { content?: unknown }).content || ""),
                });
            }
        }
        return mockComplexResult("ok");
    });

    const longQuery = "Q".repeat(750);
    try {
        await orchestrator.processQuery(longQuery, { sessionId });
    } finally {
        restore();
    }

    assert.equal(capturedHistory.length, 5);
    assert.deepEqual(
        capturedHistory.map((item) => item.content),
        ["u3", "a3", "u4", "a4", "Q".repeat(500)]
    );
});

test("phase5 orchestrator: error responses are persisted to conversation history", async () => {
    const conversationManager = new ConversationManager({
        storePath: makeStorePath("phase5_error"),
    });
    const orchestrator = new MainOrchestrator(conversationManager);
    const restore = patchExecuteComplexQuery(async () => {
        throw new Error("simulated failure");
    });

    try {
        await assert.rejects(
            orchestrator.processQuery("Trigger failure", {
                sessionId: "session-error",
            }),
            /simulated failure/
        );
    } finally {
        restore();
    }

    const persisted = await conversationManager.getConversationHistory("session-error", 10);
    assert.equal(persisted.length, 2);
    assert.equal(persisted[0].role, "user");
    assert.equal(persisted[0].content, "Trigger failure");
    assert.equal(persisted[1].role, "assistant");
    assert.ok(persisted[1].content.includes("Error: simulated failure"));
});
