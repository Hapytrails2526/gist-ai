import type { TokenCounter } from "../types.js";

/**
 * Fast, dependency-free token estimate.
 *
 * Heuristic: ~4 characters per token for English text, which tracks the
 * common BPE tokenizers (cl100k/o200k, Claude) closely enough for budgeting.
 * Swap in a real tokenizer via CompressorOptions.countTokens when you need
 * exact accounting for billing.
 */
export const estimateTokens: TokenCounter = (text: string): number => {
  if (!text) return 0;
  // Whitespace-normalize so padding doesn't inflate the count.
  const chars = text.trim().length;
  return Math.ceil(chars / 4);
};

/** Sum estimated tokens across many strings. */
export function estimateMany(
  texts: string[],
  count: TokenCounter = estimateTokens
): number {
  let total = 0;
  for (const t of texts) total += count(t);
  return total;
}
