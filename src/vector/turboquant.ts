import type { VectorCompressor } from "./compressor.js";
import { WASM_BASE64 } from "./turboquant-wasm.js";

/**
 * TurboQuant — faithful implementation of the inner-product quantizer from
 * "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
 * (Zandieh et al., arXiv:2504.19874, 2025), Algorithm 2.
 *
 * Data-oblivious: NO training pass (unlike ProductQuantizer's k-means). The
 * pipeline, per the paper:
 *
 *   1. Random rotation Π (Lemma 1): after rotating a unit vector, each
 *      coordinate follows a Beta distribution  f(x) ∝ (1−x²)^((d−3)/2), and
 *      coordinates are near-independent — so per-coordinate scalar quantization
 *      is near-optimal. We realize Π with a multi-round randomized fast
 *      Walsh–Hadamard transform (O(d·log d) "spinner"), the standard fast
 *      stand-in for a dense Haar matrix; the hot loop runs in Rust→WASM.
 *   2. MSE stage (Algorithm 1): quantize each rotated coordinate with a
 *      (b−1)-bit Lloyd–Max codebook computed for the Beta(d) density above.
 *   3. QJL stage (Definition 1): 1-bit Quantized JL on the residual,
 *      z = sign(S·r); dequant (√(π/2)/d)·Sᵀz. This yields an UNBIASED
 *      inner-product estimator (Lemma 4): E[⟨y, x̃⟩] = ⟨y, x⟩.
 *
 * Everything is kept in the Π-rotated frame so no inverse rotation is needed
 * (inner products are rotation-invariant). Stores two float norms (vector +
 * residual) plus (b−1)+1 bits per coordinate — b bits total, matching the paper.
 *
 * Deviation from the paper, documented honestly: Π and S use fast randomized
 * Hadamard transforms rather than dense Gaussian/QR matrices (O(d·log d) vs
 * O(d²)); this is the established fast realization (cf. the QJL paper) and
 * preserves the JL/distributional guarantees in high dimension.
 */
export interface TurboQuantOptions {
  /** Total bits per coordinate (2–4): (b−1) MSE bits + 1 QJL bit. Default 2. */
  bits?: 2 | 3 | 4;
  /** Rotation seed (any 32-bit). Fixed so query/stored rotations match. */
  seed?: number;
  /** Hadamard rounds for the Π rotation (Haar approximation). Default 3. */
  rounds?: number;
}

interface Kernel {
  memory: WebAssembly.Memory;
  scratch_ptr(): number;
  scratch_cap(): number;
  rotate(n: number, seed: number, scale: number): void;
  irotate(n: number, seed: number, scale: number): void;
}

let cached: Kernel | null = null;

function decodeBase64(b64: string): Uint8Array {
  if (typeof Buffer !== "undefined") return new Uint8Array(Buffer.from(b64, "base64"));
  const bin = (globalThis as { atob(s: string): string }).atob(b64);
  const u = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
  return u;
}

function loadKernel(): Kernel {
  if (cached) return cached;
  const src = decodeBase64(WASM_BASE64);
  const bytes = new Uint8Array(src.length);
  bytes.set(src);
  const mod = new WebAssembly.Module(bytes); // 614B → sync compile OK
  cached = new WebAssembly.Instance(mod, {}).exports as unknown as Kernel;
  return cached;
}

function nextPow2(n: number): number {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

/**
 * Lloyd–Max optimal centroids for the Beta coordinate density of a unit vector
 * randomly rotated in R^d:  f(x) ∝ (1−x²)^((d−3)/2) on [−1,1]. Computed
 * numerically once per (d, levels) and cached. This is the paper's MSE codebook
 * (Eq. 4) — exact for the true distribution, not just its Gaussian limit.
 */
const centroidCache = new Map<string, number[]>();
function betaCentroids(d: number, levels: number): number[] {
  const key = `${d}:${levels}`;
  const hit = centroidCache.get(key);
  if (hit) return hit;

  const exp = (d - 3) / 2;
  const range = Math.min(1, 6 / Math.sqrt(Math.max(d, 1))); // density support
  const G = 8000;
  const xs = new Array<number>(G);
  const ws = new Array<number>(G);
  for (let i = 0; i < G; i++) {
    const x = -range + (2 * range * i) / (G - 1);
    xs[i] = x;
    const base = 1 - x * x;
    ws[i] = base > 0 ? Math.pow(base, exp) : 0;
  }

  // init: evenly spaced in support
  const c: number[] = [];
  for (let k = 0; k < levels; k++) {
    c.push(-range + (2 * range * (k + 0.5)) / levels);
  }

  for (let it = 0; it < 25; it++) {
    const sum = new Array<number>(levels).fill(0);
    const wsum = new Array<number>(levels).fill(0);
    for (let i = 0; i < G; i++) {
      const x = xs[i]!;
      let best = 0;
      let bestD = Infinity;
      for (let k = 0; k < levels; k++) {
        const dd = Math.abs(x - c[k]!);
        if (dd < bestD) {
          bestD = dd;
          best = k;
        }
      }
      sum[best] = (sum[best] ?? 0) + x * ws[i]!;
      wsum[best] = (wsum[best] ?? 0) + ws[i]!;
    }
    for (let k = 0; k < levels; k++) {
      if (wsum[k]! > 0) c[k] = sum[k]! / wsum[k]!;
    }
  }
  c.sort((a, b) => a - b);
  centroidCache.set(key, c);
  return c;
}

export class TurboQuantWasm implements VectorCompressor {
  readonly name = "turboquant-wasm";
  private readonly bits: number;
  private readonly seed: number;
  private readonly rounds: number;
  private readonly kernel: Kernel;
  private readonly seedQjl: number;

  constructor(opts: TurboQuantOptions = {}) {
    this.bits = opts.bits ?? 2;
    this.seed = (opts.seed ?? 0x9e3779b9) >>> 0;
    this.rounds = opts.rounds ?? 3;
    this.seedQjl = (this.seed ^ 0x85ebca6b) >>> 0;
    this.kernel = loadKernel();
  }

  private get mseLevels(): number {
    return 1 << (this.bits - 1); // (b−1) MSE bits
  }

  /** Write src (zero-padded to `d2`) into scratch, apply `rounds` randomized
   * Hadamard rotations from `seedBase`, and return a copy of the result. */
  private rotate(src: ArrayLike<number>, len: number, d2: number, seedBase: number, rounds: number): Float32Array {
    if (d2 > this.kernel.scratch_cap()) {
      throw new Error(`TurboQuantWasm: dim ${d2} exceeds scratch capacity.`);
    }
    const view = new Float32Array(this.kernel.memory.buffer, this.kernel.scratch_ptr(), d2);
    view.fill(0);
    for (let i = 0; i < len; i++) view[i] = src[i] ?? 0;
    const scale = 1 / Math.sqrt(d2);
    for (let r = 0; r < rounds; r++) this.kernel.rotate(d2, (seedBase + r * 0x9e3779b9) >>> 0, scale);
    return view.slice(0, d2);
  }

  compress(vec: number[]): Uint8Array {
    const d2 = nextPow2(vec.length);
    let norm = 0;
    for (let i = 0; i < vec.length; i++) norm += (vec[i] ?? 0) ** 2;
    norm = Math.sqrt(norm) || 1;

    // Stage 1 — rotate, then MSE-quantize the unit-normalized coordinates.
    const yhat = this.rotate(vec, vec.length, d2, this.seed, this.rounds);
    const cents = betaCentroids(d2, this.mseLevels);
    const codes = new Uint8Array(d2);
    const resid = new Float32Array(d2);
    for (let j = 0; j < d2; j++) {
      const u = (yhat[j] ?? 0) / norm;
      let best = 0;
      let bestD = Infinity;
      for (let k = 0; k < cents.length; k++) {
        const dd = Math.abs(u - cents[k]!);
        if (dd < bestD) {
          bestD = dd;
          best = k;
        }
      }
      codes[j] = best;
      resid[j] = (yhat[j] ?? 0) - cents[best]! * norm; // residual in rotated frame
    }

    // Stage 2 — QJL: 1-bit sign of S·residual, plus residual norm.
    let rnorm = 0;
    for (let j = 0; j < d2; j++) rnorm += resid[j]! ** 2;
    rnorm = Math.sqrt(rnorm) || 0;
    const sresid = this.rotate(resid, d2, d2, this.seedQjl, 1);

    const mseBits = this.bits - 1;
    const out = new Uint8Array(8 + Math.ceil((d2 * mseBits) / 8) + Math.ceil(d2 / 8));
    const dv = new DataView(out.buffer);
    dv.setFloat32(0, norm, true);
    dv.setFloat32(4, rnorm, true);

    // pack (b−1)-bit MSE codes, then 1-bit QJL signs
    let bp = 64; // bit cursor, starting after the 8-byte header
    for (let j = 0; j < d2; j++) {
      for (let b = 0; b < mseBits; b++) {
        if ((codes[j]! >> b) & 1) out[bp >> 3]! |= 1 << (bp & 7);
        bp++;
      }
    }
    for (let j = 0; j < d2; j++) {
      if ((sresid[j] ?? 0) >= 0) out[bp >> 3]! |= 1 << (bp & 7);
      bp++;
    }
    return out;
  }

  /** Inverse of `rotate`: apply Rᵀ rounds in reverse order to undo a rotation. */
  private inverseRotate(src: ArrayLike<number>, d2: number, seedBase: number, rounds: number): Float32Array {
    const view = new Float32Array(this.kernel.memory.buffer, this.kernel.scratch_ptr(), d2);
    view.fill(0);
    for (let i = 0; i < d2; i++) view[i] = src[i] ?? 0;
    const scale = 1 / Math.sqrt(d2);
    for (let r = rounds - 1; r >= 0; r--) this.kernel.irotate(d2, (seedBase + r * 0x9e3779b9) >>> 0, scale);
    return view.slice(0, d2);
  }

  /**
   * Reconstruct an approximation of the original vector (DeQuant_prod): the MSE
   * centroids plus the QJL residual estimate, inverse-rotated back to the input
   * space. Lossy but unbiased — this is what lets TurboQuant compress VALUES
   * (reconstructed and summed), not just keys (scored). Pass the original `dim`.
   */
  decompress(compressed: Int8Array | Uint8Array, dim: number): number[] {
    const codes = compressed as Uint8Array;
    const dv = new DataView(codes.buffer, codes.byteOffset, codes.byteLength);
    const norm = dv.getFloat32(0, true);
    const rnorm = dv.getFloat32(4, true);
    const d2 = nextPow2(dim);
    const cents = betaCentroids(d2, this.mseLevels);
    const mseBits = this.bits - 1;

    // ỹ — MSE reconstruction in the rotated frame.
    const yrec = new Float32Array(d2);
    let bp = 64;
    for (let j = 0; j < d2; j++) {
      let code = 0;
      for (let b = 0; b < mseBits; b++) {
        code |= ((codes[bp >> 3]! >> (bp & 7)) & 1) << b;
        bp++;
      }
      yrec[j] = cents[code]! * norm;
    }
    // QJL residual estimate ρ̃ = (√(π/2)/d)·rnorm·Sᵀz, added in the rotated frame.
    const z = new Float32Array(d2);
    for (let j = 0; j < d2; j++) {
      z[j] = (codes[bp >> 3]! >> (bp & 7)) & 1 ? 1 : -1;
      bp++;
    }
    const sTz = this.inverseRotate(z, d2, this.seedQjl, 1);
    const qjl = (Math.sqrt(Math.PI / 2) / d2) * rnorm;
    for (let j = 0; j < d2; j++) yrec[j]! += qjl * sTz[j]!;

    // Πᵀ ŷ_rec — back to the input space; drop the zero padding.
    const xrec = this.inverseRotate(yrec, d2, this.seed, this.rounds);
    return Array.from(xrec.slice(0, dim));
  }

  similarity(query: number[], compressed: Int8Array | Uint8Array): number {
    const codes = compressed as Uint8Array;
    const dv = new DataView(codes.buffer, codes.byteOffset, codes.byteLength);
    const norm = dv.getFloat32(0, true);
    const rnorm = dv.getFloat32(4, true);

    const d2 = nextPow2(query.length);
    const cents = betaCentroids(d2, this.mseLevels);
    const mseBits = this.bits - 1;

    // Rotate the query into the same frames (full precision).
    const yq = this.rotate(query, query.length, d2, this.seed, this.rounds);
    const syq = this.rotate(yq, d2, d2, this.seedQjl, 1);

    let bp = 64;
    // term 1 — MSE reconstruction: norm · Σ yq_j · centroid(code_j)
    let t1 = 0;
    for (let j = 0; j < d2; j++) {
      let code = 0;
      for (let b = 0; b < mseBits; b++) {
        code |= ((codes[bp >> 3]! >> (bp & 7)) & 1) << b;
        bp++;
      }
      t1 += (yq[j] ?? 0) * cents[code]!;
    }
    // term 2 — QJL: (√(π/2)/d)·rnorm · Σ (S·yq)_j · sign_j
    let t2 = 0;
    for (let j = 0; j < d2; j++) {
      const sign = ((codes[bp >> 3]! >> (bp & 7)) & 1) ? 1 : -1;
      bp++;
      t2 += (syq[j] ?? 0) * sign;
    }
    return norm * t1 + (Math.sqrt(Math.PI / 2) / d2) * rnorm * t2;
  }

  bytesPer(dim: number): number {
    const d2 = nextPow2(dim);
    return 8 + Math.ceil((d2 * (this.bits - 1)) / 8) + Math.ceil(d2 / 8);
  }
}
