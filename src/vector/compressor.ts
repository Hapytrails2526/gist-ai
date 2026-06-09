/**
 * Layer 3 — embedding compression.
 *
 * This is the swappable seam. The interface is intentionally tiny so a
 * TurboQuant-style native/WASM core can drop in later behind the same shape
 * without touching MemoryStore or ContextPacker. The built-in implementations
 * here (scalar int8, binary) are real and usable today.
 */

export interface VectorCompressor {
  /** Compress a float embedding into a compact representation. */
  compress(vec: number[]): Int8Array | Uint8Array;
  /** Similarity between a query embedding and a compressed vector, 0..1-ish. */
  similarity(query: number[], compressed: Int8Array | Uint8Array): number;
  /** Bytes per stored vector, for reporting savings. */
  bytesPer(dim: number): number;
  readonly name: string;
}

/**
 * Scalar int8 quantization: map each dim from [-1,1] to [-127,127].
 * 4x smaller than float32, tiny accuracy loss for normalized embeddings.
 */
export class ScalarQuantizer implements VectorCompressor {
  readonly name = "scalar-int8";

  compress(vec: number[]): Int8Array {
    const out = new Int8Array(vec.length);
    for (let i = 0; i < vec.length; i++) {
      const clamped = Math.max(-1, Math.min(1, vec[i] ?? 0));
      out[i] = Math.round(clamped * 127);
    }
    return out;
  }

  similarity(query: number[], compressed: Int8Array | Uint8Array): number {
    const c = compressed as Int8Array;
    let dot = 0;
    for (let i = 0; i < c.length; i++)
      dot += (query[i] ?? 0) * ((c[i] ?? 0) / 127);
    return dot; // inputs are L2-normalized → this is cosine similarity
  }

  bytesPer(dim: number): number {
    return dim; // 1 byte per dim
  }
}

/**
 * Binary quantization: 1 bit per dim (sign). 32x smaller than float32.
 * Similarity via Hamming distance — coarse but extremely fast, good for a
 * first-pass shortlist before optional re-ranking with full vectors.
 */
export class BinaryQuantizer implements VectorCompressor {
  readonly name = "binary";

  compress(vec: number[]): Uint8Array {
    const out = new Uint8Array(Math.ceil(vec.length / 8));
    for (let i = 0; i < vec.length; i++) {
      if ((vec[i] ?? 0) >= 0) out[i >> 3] = (out[i >> 3] ?? 0) | (1 << (i & 7));
    }
    return out;
  }

  similarity(query: number[], compressed: Int8Array | Uint8Array): number {
    const c = compressed as Uint8Array;
    let agree = 0;
    const dim = c.length * 8;
    for (let i = 0; i < dim; i++) {
      const bit = (c[i >> 3]! >> (i & 7)) & 1;
      const qbit = (query[i] ?? 0) >= 0 ? 1 : 0;
      if (bit === qbit) agree++;
    }
    return agree / dim; // fraction of agreeing bits, 0..1
  }

  bytesPer(dim: number): number {
    return Math.ceil(dim / 8);
  }
}

/** Plain cosine similarity on full float vectors, for re-ranking/baselines. */
export function cosine(a: number[], b: number[]): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i] ?? 0;
    const y = b[i] ?? 0;
    dot += x * y;
    na += x * x;
    nb += y * y;
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
}
