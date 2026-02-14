import type { AgentContext, MCPToolResult, TaskNode } from "./types";
import { Logger } from "./logging";

export interface ValidationResult {
    isValid: boolean;
    issues: string[];
    confidence: number;
    suggestedAction?: {
        type: "retry" | "skip" | "modify";
        tool?: string;
        params?: Record<string, unknown>;
        reason?: string;
    };
    metadata: {
        validationTimestamp: number;
        validatorVersion: string;
    };
}

function metadata(): ValidationResult["metadata"] {
    return {
        validationTimestamp: Date.now(),
        validatorVersion: "1.0",
    };
}

function asRecord(value: unknown): Record<string, unknown> | null {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
        return null;
    }
    return value as Record<string, unknown>;
}

function extractContent(data: unknown): unknown {
    const asObj = asRecord(data);
    if (!asObj) {
        return data;
    }
    return asObj.content ?? data;
}

function extractText(data: unknown): string {
    if (typeof data === "string") {
        return data;
    }
    if (Array.isArray(data)) {
        return data
            .map((item) => extractText(item))
            .filter(Boolean)
            .join("\n");
    }
    const asObj = asRecord(data);
    if (!asObj) {
        return "";
    }
    const text = asObj.text;
    if (typeof text === "string") {
        return text;
    }
    return JSON.stringify(data);
}

function issueConfidence(issues: string[], penalty: number): number {
    return Math.max(0.1, 1 - (issues.length * penalty));
}

export class ValidationAgent {
    private readonly logger: Logger;

    constructor(logger: Logger) {
        this.logger = logger;
    }

    async validateToolResult(
        task: TaskNode,
        result: MCPToolResult,
        context: AgentContext
    ): Promise<ValidationResult> {
        if (!result.success) {
            const failure = {
                isValid: false,
                issues: [`Tool ${task.tool} failed: ${result.error || "unknown error"}`],
                confidence: 1.0,
                suggestedAction: {
                    type: "retry" as const,
                    tool: task.tool,
                    params: task.params,
                    reason: "Original tool call failed",
                },
                metadata: metadata(),
            };

            this.logger.warn("agent.validation.tool_failed", {
                taskId: task.id,
                tool: task.tool,
                issues: failure.issues,
            });
            return failure;
        }

        const common = this.validateCommonStructure(result);
        if (!common.isValid) {
            this.logger.warn("agent.validation.common_failed", {
                taskId: task.id,
                tool: task.tool,
                issues: common.issues,
            });
            return common;
        }

        switch (task.tool) {
            case "screen_securities":
                return this.validateScreenerResult(task, result, context);
            case "get_financial_statements":
                return this.validateFinancialsResult(task, result, context);
            case "get_technical_analysis":
                return this.validateTechnicalResult(task, result, context);
            case "get_historical_data":
                return this.validateHistoricalResult(task, result, context);
            case "get_fund_data":
                return this.validateFundResult(task, result, context);
            case "get_crypto_market":
                return this.validateCryptoResult(task, result, context);
            case "get_fx_data":
                return this.validateFxResult(task, result, context);
            case "get_sector_comparison":
                return this.validateSectorResult(task, result, context);
            default:
                return this.validateGenericResult(task, result, context);
        }
    }

    private validateCommonStructure(result: MCPToolResult): ValidationResult {
        const issues: string[] = [];
        if (!result) {
            issues.push("Result is null or undefined");
        }
        if (result && result.data === undefined) {
            issues.push("Result data is undefined");
        }

        return {
            isValid: issues.length === 0,
            issues,
            confidence: issues.length === 0 ? 1 : 0.5,
            metadata: metadata(),
        };
    }

    private async validateScreenerResult(
        task: TaskNode,
        result: MCPToolResult,
        _context: AgentContext
    ): Promise<ValidationResult> {
        void _context;
        const issues: string[] = [];
        const content = extractContent(result.data);

        if (!content) {
            issues.push("Screener returned empty data");
        } else if (Array.isArray(content) && content.length === 0) {
            issues.push("Screener returned zero matches");
        } else if (Array.isArray(content)) {
            for (let index = 0; index < Math.min(content.length, 5); index += 1) {
                const row = asRecord(content[index]);
                if (!row) {
                    issues.push(`Screener row ${index} is not an object`);
                    continue;
                }
                const symbol = row.symbol ?? row.ticker;
                if (typeof symbol !== "string" || !symbol.trim()) {
                    issues.push(`Screener row ${index} missing symbol/ticker`);
                }
            }
        }

        const suggestedAction = issues.some((issue) => issue.toLowerCase().includes("empty"))
            ? {
                type: "retry" as const,
                tool: task.tool,
                params: task.params,
                reason: "Screener returned empty results",
            }
            : undefined;

        return {
            isValid: issues.length === 0,
            issues,
            confidence: issueConfidence(issues, 0.2),
            suggestedAction,
            metadata: metadata(),
        };
    }

    private async validateFinancialsResult(
        task: TaskNode,
        result: MCPToolResult,
        _context: AgentContext
    ): Promise<ValidationResult> {
        void _context;
        const issues: string[] = [];
        const content = extractContent(result.data);
        const text = extractText(content).toLowerCase();

        if (!content) {
            issues.push("Financial statements returned empty");
        }
        if (text.includes("no financial data available")) {
            issues.push("No financial data available for symbol");
        }

        const asObj = asRecord(content);
        if (asObj) {
            for (const [key, value] of Object.entries(asObj)) {
                if (typeof value === "number" && !Number.isFinite(value)) {
                    issues.push(`Invalid numeric value in ${key}`);
                }
            }
        }

        const suggestedAction = issues.some((issue) => issue.toLowerCase().includes("empty"))
            ? {
                type: "retry" as const,
                tool: task.tool,
                params: { ...task.params, force_refresh: true },
                reason: "Financial statements returned empty",
            }
            : undefined;

        return {
            isValid: issues.length === 0,
            issues,
            confidence: issueConfidence(issues, 0.25),
            suggestedAction,
            metadata: metadata(),
        };
    }

    private async validateTechnicalResult(
        _task: TaskNode,
        result: MCPToolResult,
        _context: AgentContext
    ): Promise<ValidationResult> {
        void _task;
        void _context;
        const issues: string[] = [];
        const text = extractText(extractContent(result.data));

        if (!text.trim()) {
            issues.push("Technical analysis returned empty");
        }

        const rsiMatches = text.match(/RSI[:\s]+(\d+\.?\d*)/gi) || [];
        for (const match of rsiMatches) {
            const value = Number.parseFloat(match.split(":")[1]?.trim() || "");
            if (Number.isFinite(value) && (value < 0 || value > 100)) {
                issues.push(`Invalid RSI value: ${value}`);
            }
        }

        return {
            isValid: issues.length === 0,
            issues,
            confidence: issueConfidence(issues, 0.2),
            metadata: metadata(),
        };
    }

    private async validateHistoricalResult(
        _task: TaskNode,
        result: MCPToolResult,
        _context: AgentContext
    ): Promise<ValidationResult> {
        void _task;
        void _context;
        const issues: string[] = [];
        const content = extractContent(result.data);
        if (!content) {
            issues.push("Historical data returned empty");
        }

        const rows = Array.isArray(content) ? content : [content];
        for (let index = 0; index < Math.min(rows.length, 10); index += 1) {
            const row = asRecord(rows[index]);
            if (!row) {
                continue;
            }

            const open = typeof row.open === "number" ? row.open : undefined;
            const high = typeof row.high === "number" ? row.high : undefined;
            const low = typeof row.low === "number" ? row.low : undefined;
            const close = typeof row.close === "number" ? row.close : undefined;
            const volume = typeof row.volume === "number" ? row.volume : undefined;

            if (row.open !== undefined && open === undefined) issues.push(`Invalid open at row ${index}`);
            if (row.high !== undefined && high === undefined) issues.push(`Invalid high at row ${index}`);
            if (row.low !== undefined && low === undefined) issues.push(`Invalid low at row ${index}`);
            if (row.close !== undefined && close === undefined) issues.push(`Invalid close at row ${index}`);
            if (row.volume !== undefined && volume === undefined) issues.push(`Invalid volume at row ${index}`);

            if (high !== undefined && low !== undefined && high < low) {
                issues.push(`Row ${index} has high < low`);
            }
            if (high !== undefined && close !== undefined && close > high) {
                issues.push(`Row ${index} has close > high`);
            }
            if (low !== undefined && close !== undefined && close < low) {
                issues.push(`Row ${index} has close < low`);
            }
        }

        return {
            isValid: issues.length === 0,
            issues,
            confidence: issueConfidence(issues, 0.15),
            metadata: metadata(),
        };
    }

    private async validateFundResult(
        _task: TaskNode,
        _result: MCPToolResult,
        _context: AgentContext
    ): Promise<ValidationResult> {
        void _task;
        void _result;
        void _context;
        return {
            isValid: true,
            issues: [],
            confidence: 0.9,
            metadata: metadata(),
        };
    }

    private async validateCryptoResult(
        _task: TaskNode,
        _result: MCPToolResult,
        _context: AgentContext
    ): Promise<ValidationResult> {
        void _task;
        void _result;
        void _context;
        return {
            isValid: true,
            issues: [],
            confidence: 0.9,
            metadata: metadata(),
        };
    }

    private async validateFxResult(
        _task: TaskNode,
        _result: MCPToolResult,
        _context: AgentContext
    ): Promise<ValidationResult> {
        void _task;
        void _result;
        void _context;
        return {
            isValid: true,
            issues: [],
            confidence: 0.9,
            metadata: metadata(),
        };
    }

    private async validateSectorResult(
        _task: TaskNode,
        _result: MCPToolResult,
        _context: AgentContext
    ): Promise<ValidationResult> {
        void _task;
        void _result;
        void _context;
        return {
            isValid: true,
            issues: [],
            confidence: 0.9,
            metadata: metadata(),
        };
    }

    private async validateGenericResult(
        _task: TaskNode,
        _result: MCPToolResult,
        _context: AgentContext
    ): Promise<ValidationResult> {
        void _task;
        void _result;
        void _context;
        return {
            isValid: true,
            issues: [],
            confidence: 0.8,
            metadata: metadata(),
        };
    }
}
