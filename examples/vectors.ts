/**
 * Layer 3 benchmark — compression vs search quality.
 *
 *   npm run demo:vectors
 *
 * Builds a synthetic clustered embedding set, then for each quantizer measures
 * recall@10 against exact cosine search and the bytes per vector. Shows the
 * real trade-off: scalar is safe, binary is tiny, PQ gets binary-level size
 * with far better recall — which is why vector DBs use it.
 */
import {
  ScalarQuantizer,
  BinaryQuantizer,
  ProductQuantizer,
  TurboQuantWasm,
  cosine,
  searchWithRerank,
  type VectorCompressor,
} from "../src/index.js";

// Tiny seeded PRNG so the benchmark is reproducible.
function makeRng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 0xffffffff;
  };
}

function normalize(v: number[]): number[] {
  const n = Math.hypot(...v) || 1;
  return v.map((x) => x / n);
}

const DIM = 128;
const N = 2000;
const CLUSTERS = 25;
const TOPK = 10;
const rng = makeRng(42);

// Clustered data: each vector is a cluster center plus noise → realistic
// structure that PQ codebooks can actually learn.
const centers = Array.from({ length: CLUSTERS }, () =>
  normalize(Array.from({ length: DIM }, () => rng() * 2 - 1))
);
const data: number[][] = [];
for (let i = 0; i < N; i++) {
  const c = centers[i % CLUSTERS]!;
  data.push(normalize(c.map((x) => x + (rng() * 2 - 1) * 0.35)));
}
const query = normalize(centers[3]!.map((x) => x + (rng() * 2 - 1) * 0.2));

// Ground truth: exact cosine top-K.
const truth = new Set(
  data
    .map((v, i) => ({ i, s: cosine(query, v) }))
    .sort((a, b) => b.s - a.s)
    .slice(0, TOPK)
    .map((r) => r.i)
);

function evaluate(name: string, q: VectorCompressor): { name: string; bytes: number; recall: number } {
  const codes = data.map((v) => q.compress(v));
  const topk = codes
    .map((c, i) => ({ i, s: q.similarity(query, c) }))
    .sort((a, b) => b.s - a.s)
    .slice(0, TOPK)
    .map((r) => r.i);
  const hits = topk.filter((i) => truth.has(i)).length;
  return { name, bytes: q.bytesPer(DIM), recall: hits / TOPK };
}

console.log(`=== gist Layer 3 — ${N} vectors × ${DIM} dims, recall@${TOPK} ===\n`);
const float32 = DIM * 4;
console.log(`exact float32 baseline: ${float32} bytes/vector\n`);

const pq = new ProductQuantizer({ subvectors: 16, centroids: 256, iterations: 12 });
pq.train(data);

const rows = [
  evaluate("scalar-int8", new ScalarQuantizer()),
  evaluate("binary", new BinaryQuantizer()),
  evaluate("product-quant", pq),
];

// PQ + re-rank: use PQ for a cheap shortlist, then exact-cosine re-rank it.
const pqCodes = data.map((v) => pq.compress(v));
const reranked = searchWithRerank(query, pqCodes, data, pq, { topK: TOPK, shortlist: 64 });
const rerankHits = reranked.filter((i) => truth.has(i)).length;
rows.push({ name: "PQ + rerank", bytes: pq.bytesPer(DIM), recall: rerankHits / TOPK });

// TurboQuant (WASM) — data-oblivious, NO training. Sweep the bit-rate to show
// the rate–distortion curve the paper is actually about.
for (const b of [2, 3, 4] as const) {
  rows.push(evaluate(`turboquant b=${b}`, new TurboQuantWasm({ bits: b })));
}
const tq3 = new TurboQuantWasm({ bits: 3 });
const tqCodes = data.map((v) => tq3.compress(v));
const tqRerank = searchWithRerank(query, tqCodes, data, tq3, { topK: TOPK, shortlist: 64 });
rows.push({
  name: "TQ b=3 +rerank",
  bytes: tq3.bytesPer(DIM),
  recall: tqRerank.filter((i) => truth.has(i)).length / TOPK,
});

console.log("method         bytes   ratio    recall@10");
console.log("-------------- ------ -------- -----------");
for (const r of rows) {
  console.log(
    `${r.name.padEnd(14)} ${String(r.bytes).padStart(5)}  ${(float32 / r.bytes).toFixed(0).padStart(5)}×  ${(r.recall * 100).toFixed(0).padStart(8)}%`
  );
}
console.log(
  `\nPQ+rerank scans ${pq.bytesPer(DIM)} bytes/vec, re-ranks a 64-candidate shortlist\nagainst exact vectors — near-exact recall at a fraction of the scan cost.`
);

// Validate TurboQuant's headline property: an UNBIASED inner-product estimator
// (paper Lemma 4). similarity() IS the ⟨query, x⟩ estimate — compare to truth.
const tqIP = new TurboQuantWasm({ bits: 3 });
const ipCodes = data.map((v) => tqIP.compress(v));
const est: number[] = [];
const tru: number[] = [];
for (let i = 0; i < data.length; i++) {
  est.push(tqIP.similarity(query, ipCodes[i]!));
  tru.push(cosine(query, data[i]!)); // unit vectors ⇒ cosine = ⟨q,x⟩
}
const mean = (a: number[]) => a.reduce((s, x) => s + x, 0) / a.length;
const me = mean(est);
const mt = mean(tru);
let cov = 0;
let ve = 0;
let vt = 0;
let signedErr = 0;
for (let i = 0; i < est.length; i++) {
  const de = est[i]! - me;
  const dt = tru[i]! - mt;
  cov += de * dt;
  ve += de * de;
  vt += dt * dt;
  signedErr += est[i]! - tru[i]!;
}
console.log(`\nTurboQuant inner-product estimator (b=3, paper Lemma 4):`);
console.log(`  mean signed error : ${(signedErr / est.length).toFixed(4)}  (→0 means unbiased)`);
console.log(`  corr with true ⟨q,x⟩: ${(cov / Math.sqrt(ve * vt)).toFixed(3)}`);
