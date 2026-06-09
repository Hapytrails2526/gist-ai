import type { LLMProvider } from "../types.js";

export interface ClaudeProviderOptions {
  apiKey?: string;
  model?: string;
  /** Override base URL for proxies/gateways. */
  baseUrl?: string;
}

/**
 * Production provider backed by the Anthropic Messages API.
 *
 * Dependency-free: uses global fetch (Node 18+ / browsers / Tauri webview).
 * Reads ANTHROPIC_API_KEY from the environment if no key is passed.
 *
 * Note: the Claude API does not expose an embeddings endpoint. If you need
 * `embed`, plug in a dedicated embeddings provider (e.g. Voyage, OpenAI) —
 * gist only calls embed() when you opt into vector relevance ranking.
 */
export class ClaudeProvider implements LLMProvider {
  private readonly apiKey: string;
  private readonly model: string;
  private readonly baseUrl: string;

  constructor(opts: ClaudeProviderOptions = {}) {
    const key =
      opts.apiKey ??
      (typeof process !== "undefined" ? process.env?.ANTHROPIC_API_KEY : undefined);
    if (!key) {
      throw new Error(
        "ClaudeProvider: no API key. Pass { apiKey } or set ANTHROPIC_API_KEY."
      );
    }
    this.apiKey = key;
    this.model = opts.model ?? "claude-sonnet-4-6";
    this.baseUrl = opts.baseUrl ?? "https://api.anthropic.com";
  }

  async complete(prompt: string, opts?: { maxTokens?: number }): Promise<string> {
    const res = await fetch(`${this.baseUrl}/v1/messages`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-api-key": this.apiKey,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify({
        model: this.model,
        max_tokens: opts?.maxTokens ?? 1024,
        messages: [{ role: "user", content: prompt }],
      }),
    });

    if (!res.ok) {
      const detail = await res.text().catch(() => "");
      throw new Error(`Claude API ${res.status}: ${detail}`);
    }

    const data = (await res.json()) as {
      content?: Array<{ type: string; text?: string }>;
    };
    return (data.content ?? [])
      .filter((b) => b.type === "text")
      .map((b) => b.text ?? "")
      .join("");
  }
}
