import type { VectorCompressor } from "./compressor.js";

/**
 * Product Quantization (PQ) — the workhorse behind FAISS and most vector DBs.
 *
 * Idea: split each D-dim vector into M contiguous sub-vectors. Learn a small
 * codebook of K centroids per sub-space (k-means). Encode a vector as M bytes
 * — one centroid index per sub-space. With K ≤ 256 that's exactly M bytes,
 * independent of D. For a 1536-dim float32 embedding (6144 bytes) and M = 96
 * that's 96 bytes — a 64× reduction with strong recall.
 *
 * Similarity uses Asymmetric Distance Computation (ADC): the query stays full
 * precision and is compared against the stored centroids, so we lose far less
 * accuracy than quantizing both sides.
 *
 * Requires a one-time `train(vectors)` pass to learn the codebooks. This is the
 * stronger Layer-3 option and the natural shape a TurboQuant-style core can
 * drop into behind the same VectorCompressor interface.
 */
export interface PQOptions {
  /** Number of sub-vectors (and bytes per code). Default 8. Must divide D well. */
  subvectors?: number;
  /** Centroids per sub-space. Must be ≤ 256 (one byte per code). Default 256. */
  centroids?: number;
  /** k-means iterations during training. Default 12. */
  iterations?: number;
}

export class ProductQuantizer implements VectorCompressor {
  readonly name = "product-quant";
  private readonly M: number;
  private readonly K: number;
  private readonly iters: number;

  private dim = 0;
  private offsets: number[] = [];
  private subDims: number[] = [];
  /** codebooks[m] = K centroids, each of length subDims[m]. */
  private codebooks: number[][][] = [];
  private trained = false;

  constructor(opts: PQOptions = {}) {
    this.M = opts.subvectors ?? 8;
    this.K = opts.centroids ?? 256;
    this.iters = opts.iterations ?? 12;
    if (this.K > 256) {
      throw new Error("ProductQuantizer: centroids must be ≤ 256 (byte codes).");
    }
  }

  /** Learn codebooks from a representative sample of vectors. */
  train(vectors: number[][]): void {
    if (vectors.length === 0) throw new Error("ProductQuantizer.train: no vectors.");
    this.dim = vectors[0]!.length;

    // Split D into M contiguous sub-spaces as evenly as possible.
    this.offsets = [];
    this.subDims = [];
    const base = Math.floor(this.dim / this.M);
    let rem = this.dim % this.M;
    let off = 0;
    for (let m = 0; m < this.M; m++) {
      const size = base + (rem > 0 ? 1 : 0);
      if (rem > 0) rem--;
      this.offsets.push(off);
      this.subDims.push(size);
      off += size;
    }

    this.codebooks = [];
    for (let m = 0; m < this.M; m++) {
      const o = this.offsets[m]!;
      const s = this.subDims[m]!;
      const sub = vectors.map((v) => v.slice(o, o + s));
      const k = Math.min(this.K, sub.length);
      this.codebooks.push(kmeans(sub, k, s, this.iters));
    }
    this.trained = true;
  }

  compress(vec: number[]): Uint8Array {
    this.assertTrained();
    const codes = new Uint8Array(this.M);
    for (let m = 0; m < this.M; m++) {
      const o = this.offsets[m]!;
      const s = this.subDims[m]!;
      codes[m] = nearest(vec, o, s, this.codebooks[m]!);
    }
    return codes;
  }

  similarity(query: number[], compressed: Int8Array | Uint8Array): number {
    this.assertTrained();
    const codes = compressed as Uint8Array;
    // ADC: dot product of full-precision query against reconstructed centroids.
    // For L2-normalized inputs this approximates cosine similarity.
    let dot = 0;
    for (let m = 0; m < this.M; m++) {
      const o = this.offsets[m]!;
      const s = this.subDims[m]!;
      const centroid = this.codebooks[m]![codes[m]!]!;
      for (let i = 0; i < s; i++) dot += (query[o + i] ?? 0) * (centroid[i] ?? 0);
    }
    return dot;
  }

  bytesPer(_dim: number): number {
    return this.M; // one byte per sub-space (K ≤ 256)
  }

  private assertTrained(): void {
    if (!this.trained) throw new Error("ProductQuantizer: call train() first.");
  }
}

/** Index of the nearest centroid to vec[offset..offset+size). */
function nearest(
  vec: number[],
  offset: number,
  size: number,
  centroids: number[][]
): number {
  let best = 0;
  let bestD = Infinity;
  for (let k = 0; k < centroids.length; k++) {
    const c = centroids[k]!;
    let d = 0;
    for (let i = 0; i < size; i++) {
      const diff = (vec[offset + i] ?? 0) - (c[i] ?? 0);
      d += diff * diff;
    }
    if (d < bestD) {
      bestD = d;
      best = k;
    }
  }
  return best;
}

/**
 * Minimal Lloyd's k-means. Deterministic init (evenly-spaced seeds) so results
 * are reproducible in tests — no reliance on a global RNG.
 */
function kmeans(
  points: number[][],
  k: number,
  dim: number,
  iters: number
): number[][] {
  const n = points.length;
  const centroids: number[][] = [];
  for (let c = 0; c < k; c++) {
    centroids.push([...points[Math.floor((c * n) / k) % n]!]);
  }

  for (let it = 0; it < iters; it++) {
    const sums = centroids.map(() => new Array<number>(dim).fill(0));
    const counts = new Array<number>(k).fill(0);

    for (const p of points) {
      let best = 0;
      let bestD = Infinity;
      for (let c = 0; c < k; c++) {
        const cc = centroids[c]!;
        let d = 0;
        for (let i = 0; i < dim; i++) {
          const diff = (p[i] ?? 0) - (cc[i] ?? 0);
          d += diff * diff;
        }
        if (d < bestD) {
          bestD = d;
          best = c;
        }
      }
      counts[best]= (counts[best] ?? 0) + 1;
      const s = sums[best]!;
      for (let i = 0; i < dim; i++) s[i] = (s[i] ?? 0) + (p[i] ?? 0);
    }

    for (let c = 0; c < k; c++) {
      const cnt = counts[c] ?? 0;
      if (cnt === 0) continue; // empty cluster: leave centroid where it is
      const cc = centroids[c]!;
      const s = sums[c]!;
      for (let i = 0; i < dim; i++) cc[i] = s[i]! / cnt;
    }
  }
  return centroids;
}
