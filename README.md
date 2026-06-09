# gist

**Memory & context compression for AI applications. Keeps the gist, drops the bulk.**

`gist` is a small, dependency-free TypeScript library you drop into any app that
talks to an LLM. It shrinks what you send to the model — cutting token cost and
keeping long conversations inside the context window — without you rewriting how
your app works.

```bash
npm install gist-ai
```

```ts
import { Compressor, ClaudeProvider } from "gist-ai";

const gist = new Compressor({ provider: new ClaudeProvider(), budget: 8000 });

await gist.ingest(messages);                       // distills + stores
const ctx = await gist.buildContext({ query });    // packed, budget-aware

// send ctx.messages to your LLM; read ctx.stats for the savings.
```

The same three lines work in every app — CutMind, VoiceCAD, anything.

## What it does (the three layers)

| Layer | Component | Job |
|------|-----------|-----|
| **1. Memory** | `MemoryStore` | Keep recent turns verbatim; distill older ones into compact, deduped **facts**. |
| **2. Context** | `ContextPacker` | Given a token **budget**, assemble system + relevant memories + recent turns, trimming to fit. |
| **3. Vectors** | `VectorCompressor` | *(optional)* Quantize embeddings for fast, cheap memory recall. Swappable seam. |

Layers 1 + 2 are the default and need nothing but an LLM provider. Layer 3 is
opt-in and is where a **TurboQuant**-style native/WASM core can later drop in
behind the same tiny interface — no rework.

## Try it now (no API key)

```bash
npm install
npm run demo
```

The demo runs entirely offline with a `MockProvider` and prints the raw-vs-packed
token counts so you can see the compression ratio directly.

## Content compression (Layer 2+)

Coding agents read 10k-token logs to find one error line — and pay per token.
`gist` routes each chunk of tool output to a specialist compressor (inspired by
[Headroom](https://github.com/chopratejas/headroom)) and keeps the original
retrievable via CCR (Compress-Cache-Retrieve):

```ts
const g = new Compressor({ provider });
const r = g.compress(hugeLog);     // auto-detects type, routes, caches original
// → r.text (compressed), r.stats.saved (0..1), r.handle
const original = g.ccr.retrieve(r.handle); // nothing is ever lost
```

Measured on the demo (`npm run demo:compress`):

| content | type | tokens | saved |
|---|---|---|---|
| noisy build log | log | 1174→102 | **91%** (error line preserved) |
| repetitive JSON | json | 729→61 | **92%** (anomaly kept) |
| commented code | code | 82→64 | 22% |
| prose report | prose | 74→55 | 26% (LLMLingua-2-style pruner) |
| stack trace / test output | trace | — | **64–90%** (keeps the error + your-code frames + failing tests; collapses node_modules frames and passing tests) |
| CSV / tabular / markdown table | table | — | **85–95%** on big tables (keeps header + sample rows + anomaly rows + a count) |

Honest framing: logs/JSON compress 80–95%, code/prose 20–50% — real-world
full-session savings land ~40–50%, not the headline 95%.

### Aggressive mode

Pass `level: "aggressive"` to squeeze much harder — logs collapse to **errors +
a count**, JSON arrays fold to a **schema + count**, prose keeps fewer tokens —
while still preserving every error/anomaly. Measured by `npm run eval` (which
scores token-savings **and** signal-recall):

| | safe | aggressive | signal kept |
|---|---|---|---|
| varied log | 44% | **75%** | ✅ |
| prose | 33% | **59%** | ✅ |
| **aggregate** | 85% | **92%** | **13/13** |

```ts
g.compress(hugeLog, { level: "aggressive" }); // when you only need the gist
```

The MCP `gist_compress` tool takes the same `level` argument.

## Use it inside Claude Code / Cursor / Codex (MCP server)

`gist` ships an MCP server so any agent can compress what it reads. It exposes:

| tool | what it does |
|---|---|
| `gist_compress` | route + compress content; returns compressed text + a CCR `handle` + token stats |
| `gist_retrieve` | recover the full original for a handle (nothing is ever lost) |
| `gist_stats` | session totals — tokens in/out, saved fraction |

**Build it, then register with Claude Code:**

```bash
cd gist && npm install && npm run build
claude mcp add gist -- node "C:\Users\Theresa\gist\dist\mcp\server.js"
```

Or add to a project's `.mcp.json` (works for Cursor too):

```json
{
  "mcpServers": {
    "gist": { "command": "node", "args": ["C:\\Users\\Theresa\\gist\\dist\\mcp\\server.js"] }
  }
}
```

Verify end-to-end with `npm run test:mcp` (spawns the server, exercises all three
tools). The server is stdio, dependency-light (`@modelcontextprotocol/sdk`), and
keeps CCR originals in-process for the session's lifetime.

## Providers

- **`MockProvider`** — deterministic, offline. For tests and the demo.
- **`ClaudeProvider`** — Anthropic Messages API. Reads `ANTHROPIC_API_KEY`, or
  pass `{ apiKey }`. Dependency-free (uses `fetch`).
- **Your own** — implement the `LLMProvider` interface (`complete`, optional
  `embed`). gist is vendor-neutral.

## Integrating into a Tauri app

`gist` is plain ESM TypeScript, so it runs in the Tauri webview (frontend)
directly. Call `ingest`/`buildContext` from your UI layer; send the packed
messages to whichever model you use. The heavy vector math (Layer 3) is the only
part that may later move to the Rust backend via WASM — and it's already isolated
behind `VectorCompressor`.

## Layer 3 — quantizer benchmark

`npm run demo:vectors` (2000 vecs × 128 dims, recall@10):

| method | bytes | ratio | recall@10 | trained? |
|---|---|---|---|---|
| scalar-int8 | 128 | 4× | 100% | no |
| binary | 16 | 32× | 20% | no |
| product-quant | 16 | 32× | 50% | **yes** (k-means) |
| PQ + rerank | 16 | 32× | 100% | yes |
| turboquant b=2 | 40 | 13× | 60% | **no** |
| turboquant b=3 | 56 | 9× | 70% | **no** |
| turboquant b=4 | 72 | 7× | 80% | **no** |
| **TQ b=3 + rerank** | 56 | 9× | **100%** | no |

Recall climbs cleanly with the bit-rate (rate–distortion), and `searchWithRerank()`
recovers exact top-K from a cheap shortlist. TurboQuant's inner-product estimator
is **unbiased** (mean signed error ≈ 0, ~0.97 correlation with true ⟨q,x⟩).

### TurboQuant WASM core

`TurboQuantWasm` is a **faithful implementation of TurboQuant**
([Zandieh et al., arXiv:2504.19874](https://arxiv.org/abs/2504.19874), 2025),
Algorithm 2 — **data-oblivious** (no training pass, unlike PQ):

1. **Random rotation** → coordinates follow a Beta distribution (Lemma 1);
   realized as a multi-round randomized fast Walsh–Hadamard transform (the
   O(d·log d) hot loop, compiled **Rust → WASM**, 614 bytes, base64-inlined so it
   loads with zero plumbing in Node/browser/Tauri).
2. **MSE stage** — (b−1)-bit Lloyd–Max codebook computed for the *Beta* density.
3. **QJL stage** — 1-bit Quantized JL on the residual → an **unbiased**
   inner-product estimator (Lemma 4).

Rebuild the kernel with `npm run build:wasm` (needs `rustup target add wasm32-unknown-unknown`).

`compress()` / `decompress()` / `similarity()` form a complete codec: keys are
scored via the unbiased inner-product estimator, values are reconstructed via
`decompress()` (95% cosine fidelity at b=3) and summed.

**KV-cache demo** (`npm run demo:kvcache`) — TurboQuant's sweet spot. Both keys
and values quantized; attention output stays aligned as bits/channel drop:

| bits/channel | K+V cache (512 tok) | attn-output cosine |
|---|---|---|
| 2.5 | 40 KiB (6.4×) | 0.775 |
| 3.5 | 56 KiB (4.6×) | 0.916 |
| 4.5 | 72 KiB (3.6×) | 0.975 |

(Synthetic single head with random-Gaussian K/V — a *hard* case where attention
is near-uniform; real LLM attention is peaked, where the paper reports full
quality-neutrality at 3.5 bits/channel.)

> The one deviation from the paper: the rotation Π and JL matrix S use fast
> randomized Hadamard transforms rather than dense Gaussian/QR matrices
> (O(d·log d) vs O(d²)) — the standard fast realization, preserving the JL
> guarantees in high dimension.

## Persistence

`compressor.snapshot()` returns a JSON-serializable blob; `compressor.restore(blob)`
rehydrates it. Persist it anywhere (localStorage, a file, a DB) so memory
survives restarts.

## Roadmap

- [x] Layer 1 — tiered conversation memory + distillation
- [x] Layer 2 — budget-aware context packing with relevance ranking
- [x] Layer 3 — scalar (4×), binary (32×), and product quantization + re-rank
- [x] Persistence — snapshot / restore
- [x] Layer 3+ — faithful TurboQuant (arXiv:2504.19874) as a Rust→WASM core
- [x] Layer 2+ — content-aware compression (ContentRouter + log/json/code/prose) + CCR reversibility
- [x] LLMLingua-2-inspired prose token pruner
- [x] MCP server — `gist_compress` / `gist_retrieve` / `gist_stats` for Claude Code, Cursor, Codex
- [ ] Trained-model prose pruner (ModernBERT) + AST-aware code compression
- [ ] Layer 4 (separate module) — on-device model efficiency (BitNet/GGUF)

## Status

`v0.1.0` — working scaffold. APIs may shift before `1.0`.

MIT.
