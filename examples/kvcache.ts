/**
 * KV-cache quantization demo — TurboQuant's actual sweet spot.
 *
 *   npm run demo:kvcache
 *
 * In transformer attention, the score for each cached token is an inner product
 * ⟨q, kᵢ⟩, and the output is a softmax-weighted sum of value vectors. The KV
 * cache dominates memory at long context. TurboQuant quantizes the keys and
 * estimates those inner products *unbiasedly*, so the attention distribution —
 * and thus the output — barely moves even at ~2–3 bits per channel (the paper's
 * headline: quality-neutral at 3.5 bits, marginal loss at 2.5).
 *
 * Both keys AND values are quantized: keys are scored via the unbiased
 * inner-product estimator, values are reconstructed via decompress() and summed.
 */
import { TurboQuantWasm, cosine } from "../src/index.js";

// Seeded Gaussian RNG (Box–Muller) so the demo is reproducible.
function makeRng(seed: number) {
  let s = seed >>> 0;
  const u = () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 0xffffffff;
  };
  return () => Math.sqrt(-2 * Math.log(u() + 1e-12)) * Math.cos(2 * Math.PI * u());
}

const D = 128; // head dimension
const T = 512; // cached tokens (KV length)
const g = makeRng(7);

const K = Array.from({ length: T }, () => Array.from({ length: D }, g));
const V = Array.from({ length: T }, () => Array.from({ length: D }, g));
const q = Array.from({ length: D }, g);

function dot(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] ?? 0) * (b[i] ?? 0);
  return s;
}
function softmax(s: number[]): number[] {
  const m = Math.max(...s);
  const e = s.map((x) => Math.exp(x - m));
  const z = e.reduce((a, b) => a + b, 0) || 1;
  return e.map((x) => x / z);
}
function weightedSum(w: number[], vecs: number[][]): number[] {
  const out = new Array<number>(D).fill(0);
  for (let i = 0; i < w.length; i++) {
    const v = vecs[i]!;
    for (let j = 0; j < D; j++) out[j]! += w[i]! * (v[j] ?? 0);
  }
  return out;
}
function l2(a: number[]): number {
  return Math.sqrt(dot(a, a));
}

// Full-precision attention (ground truth).
const scale = 1 / Math.sqrt(D);
const fullScores = K.map((k) => dot(q, k) * scale);
const fullW = softmax(fullScores);
const fullOut = weightedSum(fullW, V);

console.log(`=== KV-cache quantization — 1 head, d=${D}, ${T} cached tokens ===\n`);
const fp16Bytes = D * 2 * 2; // key + value, fp16
console.log(`fp16 K+V cache: ${fp16Bytes} bytes/token · ${((fp16Bytes * T) / 1024).toFixed(0)} KiB total\n`);

interface Row {
  label: string;
  bitsCh: number;
  bytesTok: number;
  kib: number;
  outCos: number;
  relErr: number;
  kl: number;
}
const rows: Row[] = [];

let vReconCos = 0; // track value-reconstruction fidelity (b=3) for reporting
for (const bits of [2, 3, 4] as const) {
  const tq = new TurboQuantWasm({ bits });
  // Quantize keys (scored) and values (reconstructed).
  const kCodes = K.map((k) => tq.compress(k));
  const vRecon = V.map((v) => tq.decompress(tq.compress(v), D));
  const scores = kCodes.map((c) => tq.similarity(q, c) * scale);
  const w = softmax(scores);
  const out = weightedSum(w, vRecon);

  if (bits === 3) {
    let cs = 0;
    for (let i = 0; i < T; i++) cs += cosine(V[i]!, vRecon[i]!);
    vReconCos = cs / T;
  }

  // KL(full ‖ approx) over attention weights.
  let kl = 0;
  for (let i = 0; i < T; i++) {
    if (fullW[i]! > 0) kl += fullW[i]! * Math.log(fullW[i]! / Math.max(w[i]!, 1e-12));
  }
  const bytesTok = tq.bytesPer(D); // per vector; K+V cache is 2× this
  rows.push({
    label: `TurboQuant b=${bits}`,
    bitsCh: (bytesTok * 8) / D,
    bytesTok,
    kib: (bytesTok * 2 * T) / 1024, // K + V
    outCos: cosine(fullOut, out),
    relErr: l2(out.map((x, j) => x - fullOut[j]!)) / l2(fullOut),
    kl,
  });
}

console.log("method          bits/ch  bytes/tok  cache    attn-out cos  out rel-err  weight-KL");
console.log("--------------- -------  ---------  -------  ------------  -----------  ---------");
for (const r of rows) {
  console.log(
    `${r.label.padEnd(15)} ${r.bitsCh.toFixed(2).padStart(6)}  ${String(r.bytesTok).padStart(8)}  ${(r.kib.toFixed(0) + " KiB").padStart(7)}  ${r.outCos.toFixed(5).padStart(12)}  ${r.relErr.toFixed(4).padStart(11)}  ${r.kl.toExponential(2).padStart(9)}`
  );
}

const best = rows[rows.length - 1]!;
console.log(
  `\nValue reconstruction (b=3) via decompress(): ${(vReconCos * 100).toFixed(1)}% cosine vs original.`
);
console.log(
  `At ${best.bitsCh.toFixed(1)} bits/channel the full K+V cache shrinks ${(fp16Bytes / (best.bytesTok * 2)).toFixed(1)}× ` +
    `(${(fp16Bytes * T) / 1024} KiB → ${best.kib.toFixed(0)} KiB) while attention output stays ` +
    `${(best.outCos * 100).toFixed(2)}% aligned — keys scored + values reconstructed, both quantized.`
);
