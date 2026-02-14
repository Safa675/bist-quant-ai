import { executeMCPTool } from "./borsa-mcp-client";
import type { ExecutionPlan, MCPToolResult, TaskNode } from "./types";
import { Logger } from "./logging";

export interface ExecutionStats {
    startTime: number;
    endTime?: number;
    totalTasks: number;
    completedTasks: number;
    failedTasks: number;
    executionTimeMs: number;
    parallelEfficiency: number;
}

function normalizeResult(result: Awaited<ReturnType<typeof executeMCPTool>>): MCPToolResult {
    return {
        success: result.success,
        data: result.data ?? null,
        error: result.error,
        latencyMs: result.latencyMs,
    };
}

export class ParallelExecutor {
    private readonly logger: Logger;
    private readonly maxConcurrentTasks: number;

    constructor(logger: Logger, maxConcurrentTasks: number = 5) {
        this.logger = logger;
        this.maxConcurrentTasks = Math.max(1, maxConcurrentTasks);
    }

    async executeWithDependencies(
        plan: ExecutionPlan,
        requestId: string
    ): Promise<{ results: Map<string, MCPToolResult>; stats: ExecutionStats }> {
        const startTime = Date.now();
        const results = new Map<string, MCPToolResult>();
        const taskMap = new Map(plan.tasks.map((task) => [task.id, task]));
        const stats: ExecutionStats = {
            startTime,
            totalTasks: plan.tasks.length,
            completedTasks: 0,
            failedTasks: 0,
            executionTimeMs: 0,
            parallelEfficiency: 0,
        };

        this.logger.info("agent.parallel.start", {
            requestId,
            totalTasks: plan.tasks.length,
            levels: plan.executionOrder.length,
        });

        for (let levelIndex = 0; levelIndex < plan.executionOrder.length; levelIndex += 1) {
            const level = plan.executionOrder[levelIndex];
            this.logger.info("agent.parallel.level", {
                requestId,
                level: levelIndex + 1,
                levelCount: plan.executionOrder.length,
                taskCount: level.length,
                taskIds: level,
            });

            const levelResults = await this.executeLevel(level, taskMap, results, requestId);
            for (const { taskId, result } of levelResults) {
                results.set(taskId, result);
                if (result.success) {
                    stats.completedTasks += 1;
                } else {
                    stats.failedTasks += 1;
                    this.logger.warn("agent.parallel.task_failed", {
                        requestId,
                        taskId,
                        error: result.error || "unknown error",
                    });
                }
            }
        }

        stats.endTime = Date.now();
        stats.executionTimeMs = stats.endTime - stats.startTime;
        const totalPotentialParallelism = plan.executionOrder.reduce(
            (sum, level) => sum + level.length,
            0
        );
        stats.parallelEfficiency = totalPotentialParallelism > 0
            ? (stats.completedTasks + stats.failedTasks) / totalPotentialParallelism
            : 0;

        this.logger.info("agent.parallel.complete", {
            requestId,
            ...stats,
        });

        return { results, stats };
    }

    private async executeLevel(
        level: string[],
        taskMap: Map<string, TaskNode>,
        results: Map<string, MCPToolResult>,
        requestId: string
    ): Promise<Array<{ taskId: string; result: MCPToolResult }>> {
        const batches: string[][] = [];
        for (let index = 0; index < level.length; index += this.maxConcurrentTasks) {
            batches.push(level.slice(index, index + this.maxConcurrentTasks));
        }

        const allResults: Array<{ taskId: string; result: MCPToolResult }> = [];
        for (const batch of batches) {
            const batchPromises = batch.map((taskId) => this.executeSingleTask(taskId, taskMap, results, requestId));
            const batchResults = await Promise.all(batchPromises);
            allResults.push(...batchResults);
        }

        return allResults;
    }

    private async executeSingleTask(
        taskId: string,
        taskMap: Map<string, TaskNode>,
        results: Map<string, MCPToolResult>,
        requestId: string
    ): Promise<{ taskId: string; result: MCPToolResult }> {
        const task = taskMap.get(taskId);
        if (!task) {
            return {
                taskId,
                result: {
                    success: false,
                    data: null,
                    error: `Task ${taskId} not found in task map`,
                },
            };
        }

        const enrichedParams: Record<string, unknown> = { ...task.params };
        for (const depId of task.dependencies) {
            const depResult = results.get(depId);
            if (depResult?.success && depResult.data !== undefined) {
                enrichedParams[`_dep_${depId.replace(/[^a-zA-Z0-9]/g, "_")}`] = depResult.data;
            }
        }

        this.logger.debug("agent.parallel.task_start", {
            requestId,
            taskId,
            tool: task.tool,
            params: enrichedParams,
        });

        try {
            const result = normalizeResult(await executeMCPTool(
                { tool: task.tool, params: enrichedParams },
                `${requestId}-${taskId}`
            ));

            this.logger.debug("agent.parallel.task_complete", {
                requestId,
                taskId,
                success: result.success,
            });

            return { taskId, result };
        } catch (error) {
            const message = error instanceof Error ? error.message : String(error);
            this.logger.error("agent.parallel.task_exception", {
                requestId,
                taskId,
                error: message,
            });

            return {
                taskId,
                result: {
                    success: false,
                    data: null,
                    error: message,
                },
            };
        }
    }
}
