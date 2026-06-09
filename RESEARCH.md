# Memory & Compression Techniques for AI Applications — Research Brief

**Purpose:** ground and guide the `gist` library (conversation distillation, budget-aware
context packing, scalar/binary/product quantization, a faithful TurboQuant Rust→WASM
codec, and snapshot persistence).

**Method note (read this):** A multi-agent deep-research workflow gathered 23 sources and
101 candidate claims, but its *verification* stage failed mechanically (the verifier
agents never emitted structured verdicts, so every claim was auto-marked "abstain/killed"
— a tooling bug, **not** a real refutation). The facts below were therefore re-verified by
direct source fetches. Confidence is marked per claim: ✅ verified against a primary source
this pass · 🟡 from a primary source but exact figure not re-fetched · ⚠️ unverified/caveat.

_Compiled 2026-06-09._

---

## 1. TurboQuant (Google Research) ✅

**Source of record:** [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) (Zandieh &
Mirrokni); [Google Research blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/);
[OpenReview](https://openreview.net/forum?id=tO3ASKZlok).

- **It's Google.** Authored by Amir Zandieh (Research Scientist) and Vahab Mirrokni (VP,
  Google Fellow). ✅
- **Algorithm** (matches `gist`'s implementation): random rotation → coordinates follow a
  **Beta distribution** (→ Gaussian in high‑d) → per‑coordinate **Lloyd–Max optimal scalar
  quantizer**; for inner products, a two‑stage scheme = MSE quantizer **+ 1‑bit Quantized‑JL
  (QJL) on the residual**, yielding an **unbiased** inner‑product estimator. ✅
- **Data-oblivious** — zero training/fine-tuning. This is the headline differentiator vs
  product quantization. ✅
- **Distortion bound:** MSE distortion ≤ ~2.7× (√3·π/2) above the information-theoretic
  lower bound, across all bit-widths/dimensions. 🟡 (verified the ~2.7 headline; exact
  constant formula varies by metric in the paper)
- **KV-cache results:** quantizes the KV cache to **3 bits without training and without
  measurable accuracy loss**; **≥6× KV memory reduction** on long-context needle-in-haystack;
  **4-bit TurboQuant → up to 8× attention-logit throughput vs 32-bit on H100**. The paper
  states quality-neutrality at **3.5 bits/channel**, marginal degradation at **2.5**. ✅
- **vs PQ:** consistently **higher recall** than product quantization on vector search,
  with ~zero index-build time (PQ needs large codebooks + dataset-specific k-means). ✅
- **Official implementation:** none found public from Google. A third-party reimpl exists
  ([github.com/yashkc2025/turboquant](https://github.com/yashkc2025/turboquant)) — community,
  unverified, **not** official. ⚠️

**Implication for gist:** our faithful TurboQuant codec is well-aligned — keep the
data-oblivious framing as the selling point. The real sweet spot is **KV-cache (3–3.5
bits/channel)**, not generic ANN recall (where PQ+rerank already wins at lower bytes for us).
Consider exposing a 3-bit preset and documenting "training-free" prominently.

---

## 2. BitNet / BitNet b1.58 (Microsoft) ✅🟡

**Sources:** [arXiv:2504.12285](https://arxiv.org/abs/2504.12285) (BitNet b1.58 2B4T);
[arXiv:2402.17764](https://arxiv.org/html/2402.17764v1) (original b1.58).

- **BitNet b1.58 2B4T:** 2B params, trained on **4T tokens**, native 1-bit (**1.58-bit
  ternary {-1,0,+1}** weights, **8-bit** activations). **On par with leading open-weight
  full-precision LLMs of similar size** on language/math/code/chat benchmarks. Weights on
  **Hugging Face**; open-source inference for **GPU and CPU** (`bitnet.cpp`). ✅
- **Why it's efficient:** ternary weights turn matrix multiply into essentially **integer
  addition** (no FP multiply) → big energy savings. ✅
- **Memory:** ~**0.4 GB non-embedding** vs 2 GB+ for comparable FP models (~5× reduction). 🟡
- **Original b1.58 (3B):** matched FP16 LLaMA perplexity (**9.91 vs 10.04**) while **2.71×
  faster** and using **3.55× less GPU memory**. 🟡

**Implication for gist:** BitNet is **Layer 4** — a *model-runtime* concern (running a
ternary model), not a memory/context library task. Correctly kept as a **separate future
module**, not part of the gist codec. If pursued, integrate via `bitnet.cpp` in a Tauri
backend, not in the TS library.

---

## 3. LLMLingua / LLMLingua-2 (Microsoft) ✅

**Sources:** [github.com/microsoft/LLMLingua](https://github.com/microsoft/LLMLingua);
[LLMLingua-2, ACL 2024 Findings](https://aclanthology.org/2024.findings-acl.57/).

- **LLMLingua:** up to **20× prompt compression** with minimal performance loss; uses a
  small LM (GPT2-small / LLaMA-7B) to score and **drop non-essential tokens**. EMNLP 2023. ✅
- **LongLLMLingua:** **+21.4% RAG performance using ~1/4 of the tokens.** ✅
- **LLMLingua-2:** **token classification with a BERT-level encoder**, trained by data
  distillation from GPT-4; **3–6× faster** than LLMLingua and better on out-of-domain text;
  task-agnostic. ACL 2024. ✅

**Implication for gist:** this is the concrete upgrade for **Layer 2 (ContextPacker)**. Our
packer currently keeps/drops whole turns; an LLMLingua-2-style **token-level pruner** (a
small classifier scoring token salience) would push compression well past our current ~3×.
Highest-value next feature for the context layer.

---

## 4. Agent long-term memory — mem0 & MemGPT/Letta ✅🟡

**Sources:** [mem0.ai/research](https://mem0.ai/research);
[github.com/mem0ai/mem0](https://github.com/mem0ai/mem0).

- **mem0 mechanism:** **extraction** (single-pass, ADD-only — agent-confirmed facts as
  primary data) + **multi-signal retrieval** (parallel semantic + keyword + entity scoring).
  This is essentially what `gist`'s MemoryStore does (distill → store facts). ✅
- **mem0 benchmarks (verified):** **LOCOMO 92.5** overall (1,540 Qs); **< 7,000 tokens per
  retrieval** vs **25,000+** for full-context (~**3–4× token reduction**); LongMemEval 94.4. ✅
- **Commonly-cited paper headline** (vs OpenAI memory: +26% accuracy, ~91% lower p95 latency,
  ~90% token savings) — **not re-confirmed this pass**; the research page now shows the
  updated LOCOMO numbers above instead. ⚠️
- **MemGPT/Letta:** OS-style tiered memory (in-context "main" + external archival, paged in
  via function calls). Named-baseline head-to-head numbers not re-verified here. 🟡

**Implication for gist:** validates our Layer-1 design (distill to durable facts, retrieve
by relevance). Two concrete upgrades: (a) **multi-signal retrieval** (we do semantic only via
embeddings — add keyword + entity matching), and (b) an **ADD/UPDATE/consolidate** step so
facts get merged/superseded, not just deduped.

---

## 5. Embedding quantization for retrieval ✅

**Source:** [Hugging Face — Embedding Quantization](https://huggingface.co/blog/embedding-quantization).

| Method | Compression | Recall retention | Speed |
|---|---|---|---|
| **int8 / scalar** | **4×** | ~**99.3%** (99% w/ rescoring) | ~3.66× avg |
| **binary** | **32×** | ~92.5% raw, **~96% w/ rescoring** | **24.76× avg** (up to 45.8×) |
| **Matryoshka (MRL)** | e.g. 12× | 93.1% (OpenAI 3-large @12×); 90% @6× (Nomic) | — |

- **Rescoring is the unlock:** retrieve with binary codes, then re-rank top candidates with
  full-precision query embeddings → recovers binary from 92.5% → **96.45%** (mxbai example). ✅
- **Combined binary+int8 rescoring demo:** 5 GB RAM + 50 GB disk vs **200 GB** for float32. ✅
- MRL (truncatable embeddings) is **orthogonal** to quantization — they stack. ✅

**Implication for gist:** this is **exactly our Layer-3 `searchWithRerank()` pattern** —
quantize, shortlist, re-rank with exact vectors. Validated externally. Two adds worth doing:
(a) a **Matryoshka-style truncation** option (cheap dimensionality cut, stacks with our
quantizers), and (b) document the binary+rerank path as the cheapest high-recall config
(our PQ+rerank already hits 100% recall@10 at 16 bytes).

---

## "Nola AI"? — No. ⚠️→resolved

There is **no verifiable company/product called "Nola AI"** in the memory/compression space.
The only close match is **NOLA** ([arXiv:2310.02556](https://arxiv.org/pdf/2310.02556)) —
an academic method, *"Compressing LoRA using Linear Combination of Random Basis"* — which is
about **parameter-efficient fine-tuning** compression, unrelated to vector/KV/context
compression. Treat "Nola" as available for **your own branding** (the library is `gist`).

---

## Most actionable findings (TL;DR)

1. **TurboQuant = Google, data-oblivious, KV-cache @3-bit, training-free.** Our codec matches
   it; lean into the KV-cache + training-free story, add a 3-bit preset.
2. **Biggest gist upgrade: LLMLingua-2-style token pruning in Layer 2** — pushes context
   compression far past whole-turn dropping.
3. **Layer 1: add multi-signal retrieval + ADD/UPDATE/consolidate** (mem0 pattern).
4. **Layer 3 is externally validated** (HF confirms quantize→rescore); add Matryoshka
   truncation as a stacking option.
5. **BitNet stays Layer 4** (model runtime, separate module) — don't fold into the TS library.
6. **"Nola AI" isn't real** — it's yours to claim.

---

## Sources

- TurboQuant: [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) ·
  [HTML](https://arxiv.org/html/2504.19874v1) ·
  [Google Research blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) ·
  [OpenReview](https://openreview.net/forum?id=tO3ASKZlok)
- BitNet: [arXiv:2504.12285 (b1.58 2B4T)](https://arxiv.org/abs/2504.12285) ·
  [arXiv:2402.17764 (b1.58)](https://arxiv.org/html/2402.17764v1)
- LLMLingua: [github.com/microsoft/LLMLingua](https://github.com/microsoft/LLMLingua) ·
  [LLMLingua-2 (ACL 2024)](https://aclanthology.org/2024.findings-acl.57/)
- Agent memory: [mem0.ai/research](https://mem0.ai/research) ·
  [github.com/mem0ai/mem0](https://github.com/mem0ai/mem0)
- Embedding quantization: [Hugging Face blog](https://huggingface.co/blog/embedding-quantization)
- "Nola"/NOLA disambiguation: [arXiv:2310.02556](https://arxiv.org/pdf/2310.02556)
