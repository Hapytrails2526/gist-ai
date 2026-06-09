import type { VectorCompressor } from "./compressor.js";
import { cosine } from "./compressor.js";

/**
 * Two-stage search: fast approximate shortlist, then exact re-rank.
 *
 * Stage 1 scans the compressed codes (cheap, cache-friendly) to gather a
 * shortlist of `shortlist` candidates. Stage 2 re-scores ONLY those candidates
 * with exact cosine against the original float vectors and returns the top
 * `topK`. This recovers almost all the recall a lossy quantizer (binary, PQ)
 * gives up, while still scanning the whole set cheaply — the standard pattern
 * in production vector search.
 *
 * You pay the float-vector storage only if you want re-rank; the shortlist
 * scan itself needs just the codes.
 */
export function searchWithRerank(
  query: number[],
  codes: Array<Int8Array | Uint8Array>,
  vectors: number[][],
  quantizer: VectorCompressor,
  opts: { topK?: number; shortlist?: number } = {}
): number[] {
  const topK = opts.topK ?? 10;
  const shortlist = Math.max(opts.shortlist ?? topK * 6, topK);

  // Stage 1 — approximate scan over compressed codes.
  const candidates = codes
    .map((c, i) => ({ i, s: quantizer.similarity(query, c) }))
    .sort((a, b) => b.s - a.s)
    .slice(0, shortlist)
    .map((r) => r.i);

  // Stage 2 — exact re-rank of the shortlist only.
  return rerankByCosine(query, candidates, vectors, topK);
}

/** Re-score the given candidate indices with exact cosine, return top-K. */
export function rerankByCosine(
  query: number[],
  candidateIdxs: number[],
  vectors: number[][],
  topK: number
): number[] {
  return candidateIdxs
    .map((i) => ({ i, s: cosine(query, vectors[i]!) }))
    .sort((a, b) => b.s - a.s)
    .slice(0, topK)
    .map((r) => r.i);
}
