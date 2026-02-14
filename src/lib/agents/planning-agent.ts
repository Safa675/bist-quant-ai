import type { AgentContext, ExecutionPlan, TaskNode } from "./types";
import { callAzureOpenAI } from "../utils/azure-openai";

const DEFAULT_TASK_PRIORITY = 3;

function sanitizeTask(raw: Partial<TaskNode>, index: number): TaskNode {
    const id = String(raw.id || `task_${index + 1}`).trim() || `task_${index + 1}`;
    const dependencies = Array.isArray(raw.dependencies)
        ? raw.dependencies.map((dep) => String(dep).trim()).filter(Boolean)
        : [];
    const priorityCandidate = Number(raw.priority);
    const priority = Number.isFinite(priorityCandidate)
        ? Math.max(1, Math.min(5, Math.trunc(priorityCandidate)))
        : DEFAULT_TASK_PRIORITY;

    return {
        id,
        description: String(raw.description || `Task ${index + 1}`).trim(),
        tool: String(raw.tool || "").trim(),
        params: (raw.params && typeof raw.params === "object")
            ? (raw.params as Record<string, unknown>)
            : {},
        dependencies,
        status: "pending",
        result: raw.result,
        priority,
    };
}

export async function createExecutionPlan(
    query: string,
    context: AgentContext
): Promise<ExecutionPlan> {
    const planningPrompt = `You are a financial analysis planning agent.

Given the user query, decompose it into atomic tasks that can be executed
using the available Borsa MCP tools. For each task, specify:
1. A unique task ID
2. Which tool to call
3. The parameters for the tool
4. Which other tasks it depends on (by ID)
5. Priority level (1-5, where 5 is highest priority)

Consider dependencies carefully.

Available tools:
- search_symbol
- get_profile
- get_quick_info
- get_historical_data
- get_technical_analysis
- get_pivot_points
- get_analyst_data
- get_dividends
- get_earnings
- get_financial_statements
- get_financial_ratios
- get_corporate_actions
- get_news
- screen_securities
- scan_stocks
- get_crypto_market
- get_fx_data
- get_economic_calendar
- get_bond_yields
- get_sector_comparison
- get_fund_data
- get_index_data
- get_macro_data

Context:
${JSON.stringify(context, null, 2)}

User Query: ${query}

Respond ONLY with a JSON object:
{
  "tasks": [
    {
      "id": "unique_task_id",
      "description": "Brief description of what this task does",
      "tool": "tool_name",
      "params": {"param1": "value1"},
      "dependencies": ["dependency_task_id"],
      "priority": 3
    }
  ]
}`;

    const response = await callAzureOpenAI(planningPrompt, {
        temperature: 0.1,
        max_tokens: 2000,
        response_format: { type: "json_object" },
        requestId: context.requestId,
    });

    let parsedResponse: unknown = null;
    try {
        parsedResponse = JSON.parse(response);
    } catch {
        throw new Error(`Planning agent returned invalid JSON: ${response.slice(0, 300)}`);
    }

    const rawTasks = (parsedResponse as { tasks?: unknown })?.tasks;
    if (!Array.isArray(rawTasks) || rawTasks.length === 0) {
        throw new Error("Planning agent returned no tasks.");
    }

    const tasks = rawTasks
        .map((task, index) => sanitizeTask(task as Partial<TaskNode>, index))
        .filter((task) => Boolean(task.tool));

    if (tasks.length === 0) {
        throw new Error("Planning agent produced no executable tasks.");
    }

    const executionOrder = topologicalSort(tasks);
    return { tasks, executionOrder };
}

export function topologicalSort(tasks: TaskNode[]): string[][] {
    const inDegree = new Map<string, number>();
    const graph = new Map<string, string[]>();
    const taskMap = new Map(tasks.map((task) => [task.id, task]));

    for (const task of tasks) {
        inDegree.set(task.id, task.dependencies.length);
        graph.set(task.id, []);
    }

    for (const task of tasks) {
        for (const dep of task.dependencies) {
            if (!taskMap.has(dep)) {
                throw new Error(`Dependency ${dep} not found in tasks`);
            }
            const edges = graph.get(dep) || [];
            edges.push(task.id);
            graph.set(dep, edges);
        }
    }

    const levels: string[][] = [];
    let queue = tasks
        .filter((task) => task.dependencies.length === 0)
        .sort((a, b) => b.priority - a.priority)
        .map((task) => task.id);

    while (queue.length > 0) {
        const currentLevel = [...queue].sort((a, b) => {
            const taskA = taskMap.get(a);
            const taskB = taskMap.get(b);
            if (!taskA || !taskB) return 0;
            return taskB.priority - taskA.priority;
        });
        levels.push(currentLevel);

        const nextQueue: string[] = [];
        for (const nodeId of queue) {
            for (const neighbor of graph.get(nodeId) || []) {
                const newDegree = (inDegree.get(neighbor) || 1) - 1;
                inDegree.set(neighbor, newDegree);
                if (newDegree === 0) {
                    nextQueue.push(neighbor);
                }
            }
        }
        queue = nextQueue.sort((a, b) => {
            const taskA = taskMap.get(a);
            const taskB = taskMap.get(b);
            if (!taskA || !taskB) return 0;
            return taskB.priority - taskA.priority;
        });
    }

    if (levels.flat().length !== tasks.length) {
        throw new Error("Cycle detected in task dependencies");
    }

    return levels;
}
