import { logError } from "@/lib/agents/logging";

interface AzureOpenAIConfig {
    endpoint: string;
    apiKey: string;
    deployment: string;
    apiVersion: string;
}

export interface AzureOpenAICallOptions {
    temperature?: number;
    max_tokens?: number;
    response_format?: { type: "json_object" | "text" };
    systemPrompt?: string;
    requestId?: string;
}

const REQUIRED_AZURE_ENV = [
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_DEPLOYMENT",
] as const;

function env(name: string): string {
    return (process.env[name] || "").trim();
}

function getAzureConfig(): AzureOpenAIConfig {
    const missing = REQUIRED_AZURE_ENV.filter((name) => !env(name));
    if (missing.length > 0) {
        throw new Error(
            `Missing required Azure OpenAI environment variable(s): ${missing.join(", ")}.`
        );
    }

    return {
        endpoint: env("AZURE_OPENAI_ENDPOINT").replace(/\/$/, ""),
        apiKey: env("AZURE_OPENAI_API_KEY"),
        deployment: env("AZURE_OPENAI_DEPLOYMENT"),
        apiVersion: env("AZURE_OPENAI_API_VERSION") || "2024-10-21",
    };
}

function extractAssistantText(payload: unknown): string {
    const message = (payload as {
        choices?: Array<{
            message?: {
                content?: string | Array<{ text?: string } | string> | null;
            };
        }>;
    })?.choices?.[0]?.message;

    const content = message?.content;
    if (typeof content === "string") {
        return content.trim();
    }
    if (Array.isArray(content)) {
        return content
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
    }
    return "";
}

export async function callAzureOpenAI(
    prompt: string,
    options: AzureOpenAICallOptions = {}
): Promise<string> {
    const cfg = getAzureConfig();
    const url = `${cfg.endpoint}/openai/deployments/${encodeURIComponent(cfg.deployment)}/chat/completions` +
        `?api-version=${encodeURIComponent(cfg.apiVersion)}`;

    const response = await fetch(url, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "api-key": cfg.apiKey,
        },
        body: JSON.stringify({
            messages: [
                ...(options.systemPrompt ? [{ role: "system", content: options.systemPrompt }] : []),
                { role: "user", content: prompt },
            ],
            temperature: options.temperature ?? 0.2,
            max_completion_tokens: options.max_tokens ?? 1200,
            response_format: options.response_format,
        }),
        cache: "no-store",
    });

    const raw = await response.text();
    let parsed: unknown = null;
    try {
        parsed = JSON.parse(raw);
    } catch {
        throw new Error(`Invalid Azure OpenAI response JSON: ${raw.slice(0, 300)}`);
    }

    if (!response.ok) {
        const msg = (parsed as { error?: { message?: string } })?.error?.message || raw.slice(0, 300);
        const error = new Error(`Azure OpenAI error (${response.status}): ${msg}`);
        logError("azure.openai.call.failed", {
            requestId: options.requestId || null,
            status: response.status,
            error: error.message,
        });
        throw error;
    }

    const text = extractAssistantText(parsed);
    if (!text) {
        throw new Error("Azure OpenAI returned an empty assistant response.");
    }
    return text;
}
