import type {
  LLMProvider,
  Memory,
  Message,
  PackedContext,
  TokenCounter,
} from "../types.js";
import { cosine } from "../vector/compressor.js";
import { estimateTokens } from "./tokens.js";

export interface PackOptions {
  budget: number;
  countTokens?: TokenCounter;
  /** Optional query to rank memories by relevance (needs provider.embed). */
  query?: string;
  provider?: LLMProvider;
  /** System preamble that is always included (instructions, persona). */
  system?: string;
  /**
   * True uncompressed baseline (every turn ever ingested) for the stats.
   * When omitted, falls back to the tokens still held in `recent` + memories,
   * which understates savings once distillation has discarded originals.
   */
  rawTokens?: number;
}

/**
 * Layer 2 — assemble a prompt that fits a token budget.
 *
 * Priority order, highest first:
 *   1. system preamble (always kept)
 *   2. most-recent verbatim turns (newest first)
 *   3. distilled memories, ranked by relevance to `query` when available
 *
 * Whatever doesn't fit is dropped, lowest-priority first. The returned stats
 * compare the packed size against the full raw history so you can see the
 * compression ratio directly.
 */
export async function packContext(
  recent: Message[],
  memories: Memory[],
  opts: PackOptions
): Promise<PackedContext> {
  const count = opts.countTokens ?? estimateTokens;
  const rawTokens =
    opts.rawTokens ??
    recent.reduce((s, m) => s + count(m.content), 0) +
      memories.reduce((s, m) => s + count(m.text), 0);

  let remaining = opts.budget;
  const picked: Message[] = [];

  // 1. system preamble
  if (opts.system) {
    const t = count(opts.system);
    if (t <= remaining) {
      picked.push({ role: "system", content: opts.system });
      remaining -= t;
    }
  }

  // 2. recent verbatim turns, newest first
  const keptRecent: Message[] = [];
  for (let i = recent.length - 1; i >= 0; i--) {
    const m = recent[i]!;
    const t = count(m.content);
    if (t <= remaining) {
      keptRecent.unshift(m);
      remaining -= t;
    }
  }

  // 3. memories, relevance-ranked when we have an embedder + query
  let ranked = memories;
  if (opts.query && opts.provider?.embed) {
    const [q] = await opts.provider.embed([opts.query]);
    const memVecs = await opts.provider.embed(memories.map((m) => m.text));
    ranked = memories
      .map((m, i) => ({ m, score: q ? cosine(q, memVecs[i]!) : 0 }))
      .sort((a, b) => b.score - a.score)
      .map((r) => r.m);
  }

  const memLines: string[] = [];
  let memoriesUsed = 0;
  for (const m of ranked) {
    const t = count(m.text) + 2;
    if (t <= remaining) {
      memLines.push(`- ${m.text}`);
      remaining -= t;
      memoriesUsed++;
    }
  }
  if (memLines.length > 0) {
    picked.unshift({
      role: "system",
      content: `Relevant memory:\n${memLines.join("\n")}`,
    });
  }

  // recent turns come after the memory block
  picked.push(...keptRecent);

  const packedTokens = picked.reduce((s, m) => s + count(m.content), 0);
  return {
    messages: picked,
    stats: {
      rawTokens,
      packedTokens,
      ratio: packedTokens > 0 ? +(rawTokens / packedTokens).toFixed(2) : 1,
      rawMessages: recent.length,
      packedMessages: picked.length,
      memoriesUsed,
    },
  };
}
