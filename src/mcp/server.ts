#!/usr/bin/env node
/**
 * gist MCP server — content compression as tools for coding agents.
 *
 * Exposes gist's Layer-2 content compression over MCP (stdio) so Claude Code,
 * Cursor, Codex, etc. can shrink the logs / JSON / code / prose their tools
 * read BEFORE it hits the model — with CCR reversibility so nothing is lost.
 *
 * Tools: gist_compress · gist_retrieve · gist_stats
 */
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { compressContent, type ContentType } from "../compress/router.js";
import { ReversibleStore } from "../compress/ccr.js";

// Session state — persists across tool calls within one server process.
const store = new ReversibleStore();
const session = { calls: 0, tokensIn: 0, tokensOut: 0 };

const CHARACTER_LIMIT = 100_000; // guard against pathologically large retrieves

const server = new McpServer({ name: "gist-mcp-server", version: "0.1.0" });

server.registerTool(
  "gist_compress",
  {
    title: "Compress content (gist)",
    description: `Compress a chunk of tool output before it goes to the model, routing by content type to a specialist compressor. Reduces token cost while preserving the signal (errors, anomalies, key facts). The original is cached and a retrieve-handle is returned, so nothing is permanently lost.

Args:
  - content (string): the raw text to compress (a log, JSON blob, source file, or prose)
  - type ('auto'|'log'|'json'|'code'|'prose'): force a compressor, or 'auto' to detect (default 'auto')
  - level ('safe'|'aggressive'): 'aggressive' squeezes much harder (logs→errors only, JSON→schema+count) while still keeping errors/anomalies. Use when you only need the gist. Default 'safe'.
  - keep (number 0.05–1): for prose, the fraction of tokens to keep (default 0.5, or 0.3 when aggressive)

Returns structured data:
  {
    "type": string,              // compressor used
    "handle": string,            // CCR handle — pass to gist_retrieve to recover the original
    "originalTokens": number,
    "compressedTokens": number,
    "saved": number,             // fraction saved, 0..1
    "ratio": number              // original/compressed
  }
The text content IS the compressed output (use it directly). Logs/JSON typically save 80–95%; code/prose 20–50%.

Use when: a tool returned a large log/JSON/file you only need the gist of.
Don't use when: the content is already small (< ~200 tokens) — overhead isn't worth it.`,
    inputSchema: {
      content: z.string().min(1, "content must not be empty").describe("Raw text to compress"),
      type: z
        .enum(["auto", "log", "json", "code", "prose", "trace", "table"])
        .default("auto")
        .describe("Force a compressor or auto-detect ('trace' = stack traces / test output; 'table' = CSV / tabular)"),
      level: z
        .enum(["safe", "aggressive"])
        .default("safe")
        .describe("'aggressive' squeezes harder; still keeps errors/anomalies"),
      keep: z
        .number()
        .min(0.05)
        .max(1)
        .optional()
        .describe("Prose: fraction of tokens to keep (default 0.5)"),
    },
    annotations: {
      readOnlyHint: false, // caches the original in the CCR store
      destructiveHint: false,
      idempotentHint: false, // each call mints a new handle
      openWorldHint: false,
    },
  },
  async ({ content, type, level, keep }) => {
    const r = compressContent(content, {
      type: type === "auto" ? undefined : (type as ContentType),
      level,
      keep,
      reversible: store,
    });
    session.calls += 1;
    session.tokensIn += r.stats.originalTokens;
    session.tokensOut += r.stats.compressedTokens;

    const output = {
      type: r.type,
      handle: r.handle,
      originalTokens: r.stats.originalTokens,
      compressedTokens: r.stats.compressedTokens,
      saved: r.stats.saved,
      ratio: r.stats.ratio,
    };
    return {
      content: [{ type: "text" as const, text: r.text }],
      structuredContent: output,
    };
  }
);

server.registerTool(
  "gist_retrieve",
  {
    title: "Retrieve original (gist)",
    description: `Recover the full, uncompressed original for a handle returned by gist_compress. Use this when the compressed version dropped a detail you now need.

Args:
  - handle (string): the CCR handle (e.g. "ccr_3") from a prior gist_compress call

Returns the original text. Errors with guidance if the handle is unknown (it may be from a previous session — handles live only for the current server process).`,
    inputSchema: {
      handle: z
        .string()
        .regex(/^ccr_[a-z0-9]+$/, "handle must look like 'ccr_<id>'")
        .describe("CCR handle from a prior gist_compress call"),
    },
    annotations: {
      readOnlyHint: true,
      destructiveHint: false,
      idempotentHint: true,
      openWorldHint: false,
    },
  },
  async ({ handle }) => {
    const original = store.retrieve(handle);
    if (original === undefined) {
      return {
        content: [
          {
            type: "text" as const,
            text: `Error: unknown handle '${handle}'. Handles are only valid within the current server session and are minted by gist_compress. Re-compress the source to get a fresh handle.`,
          },
        ],
        isError: true,
      };
    }
    let text = original;
    let truncated = false;
    if (text.length > CHARACTER_LIMIT) {
      text = text.slice(0, CHARACTER_LIMIT);
      truncated = true;
    }
    return {
      content: [
        {
          type: "text" as const,
          text: truncated
            ? `${text}\n\n…(truncated from ${original.length} chars; raise CHARACTER_LIMIT if you need the full original)`
            : text,
        },
      ],
      structuredContent: { handle, length: original.length, truncated },
    };
  }
);

server.registerTool(
  "gist_stats",
  {
    title: "Compression stats (gist)",
    description: `Report this session's cumulative compression savings.

Returns: { "calls": number, "tokensIn": number, "tokensOut": number, "tokensSaved": number, "savedFraction": number, "storedOriginals": number }`,
    inputSchema: {},
    annotations: {
      readOnlyHint: true,
      destructiveHint: false,
      idempotentHint: true,
      openWorldHint: false,
    },
  },
  async () => {
    const saved = session.tokensIn - session.tokensOut;
    const output = {
      calls: session.calls,
      tokensIn: session.tokensIn,
      tokensOut: session.tokensOut,
      tokensSaved: saved,
      savedFraction: session.tokensIn > 0 ? +(saved / session.tokensIn).toFixed(3) : 0,
      storedOriginals: store.size,
    };
    return {
      content: [{ type: "text" as const, text: JSON.stringify(output, null, 2) }],
      structuredContent: output,
    };
  }
);

async function main(): Promise<void> {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  // stdout is the protocol channel — log to stderr only.
  console.error("gist-mcp-server running on stdio (tools: gist_compress, gist_retrieve, gist_stats)");
}

main().catch((err) => {
  console.error("gist-mcp-server fatal error:", err);
  process.exit(1);
});
