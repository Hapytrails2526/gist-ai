/**
 * Integration test for the gist MCP server.
 *
 *   npm run build && npm run test:mcp
 *
 * Spawns the built server over stdio with a real MCP client, then exercises
 * gist_compress → gist_retrieve → gist_stats end-to-end.
 */
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

interface ToolResult {
  content: Array<{ type: string; text?: string }>;
  structuredContent?: Record<string, unknown>;
  isError?: boolean;
}

async function main(): Promise<void> {
  const transport = new StdioClientTransport({ command: "node", args: ["dist/mcp/server.js"] });
  const client = new Client({ name: "gist-mcp-test", version: "0.0.0" });
  await client.connect(transport);

  const tools = await client.listTools();
  console.log("tools exposed:", tools.tools.map((t) => t.name).join(", "));

  const log = [
    ...Array.from({ length: 30 }, (_, i) => `2026-06-09T10:00:${String(i).padStart(2, "0")}Z INFO ok job ${i}`),
    "2026-06-09T10:00:30Z ERROR boom: ECONNREFUSED 10.0.0.5:5432",
  ].join("\n");

  const c = (await client.callTool({
    name: "gist_compress",
    arguments: { content: log, type: "log" },
  })) as ToolResult;
  const sc = c.structuredContent as { handle: string; saved: number; originalTokens: number; compressedTokens: number };
  console.log(`compress: ${sc.originalTokens}→${sc.compressedTokens} tok (${Math.round(sc.saved * 100)}% saved), handle=${sc.handle}`);
  console.log("  error preserved in output:", (c.content[0]?.text ?? "").includes("ECONNREFUSED") ? "✓" : "✗");

  const r = (await client.callTool({
    name: "gist_retrieve",
    arguments: { handle: sc.handle },
  })) as ToolResult;
  console.log("retrieve returns exact original:", r.content[0]?.text === log ? "✓" : "✗");

  const bad = (await client.callTool({
    name: "gist_retrieve",
    arguments: { handle: "ccr_zzz" },
  })) as ToolResult;
  console.log("unknown handle is a clean error:", bad.isError ? "✓" : "✗");

  const s = (await client.callTool({ name: "gist_stats", arguments: {} })) as ToolResult;
  console.log("stats:", JSON.stringify(s.structuredContent));

  await client.close();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
