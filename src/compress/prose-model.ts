/**
 * Trained-model prose pruner — LLMLingua(-1)-style perplexity pruning.
 *
 * Loads a small causal LM via transformers.js (ONNX) and scores each token by
 * SURPRISAL (−log p(token | left context)). Predictable, low-surprisal tokens
 * carry little information and are dropped; surprising, high-surprisal tokens
 * are kept. This is LLMLingua's original signal, realized with a real trained
 * model instead of the heuristic in `prose.ts`.
 *
 * OPTIONAL: install `@huggingface/transformers` to use it — it is deliberately
 * NOT a dependency of the core library (which stays dependency-free). The first
 * call downloads the model (~tens of MB quantized) and caches it.
 *
 * Honest note on "ModernBERT": LLMLingua-2's token-classification variant
 * (BERT/XLM-R) would score keep/drop even better, but its weights are not
 * published as ONNX for transformers.js. This perplexity approach uses a model
 * (default `Xenova/distilgpt2`) that is. Swap `model` for any ONNX causal LM.
 */
export interface ModelProsePrunerOptions {
  /** transformers.js causal-LM model id with an ONNX export. */
  model?: string;
  /** Quantized weights (smaller/faster). Default true. */
  quantized?: boolean;
}

export class ModelProsePruner {
  private model: unknown = null;
  private tokenizer: unknown = null;
  private readonly modelId: string;
  private readonly quantized: boolean;

  constructor(opts: ModelProsePrunerOptions = {}) {
    this.modelId = opts.model ?? "Xenova/distilgpt2";
    this.quantized = opts.quantized ?? true;
  }

  /** Lazily load the model + tokenizer (downloads on first call). */
  async init(): Promise<void> {
    if (this.model) return;
    // Computed specifier so the core typechecks without the optional dep.
    const spec = "@huggingface/transformers";
    let t: any;
    try {
      t = await import(/* @vite-ignore */ spec);
    } catch {
      throw new Error(
        "ModelProsePruner requires '@huggingface/transformers'. Install it with: npm i @huggingface/transformers"
      );
    }
    this.tokenizer = await t.AutoTokenizer.from_pretrained(this.modelId);
    this.model = await t.AutoModelForCausalLM.from_pretrained(this.modelId, {
      dtype: this.quantized ? "q8" : "fp32",
    });
  }

  /**
   * Keep ~`keep` fraction of tokens, dropping the most predictable ones.
   * Returns information-dense (not fluent) text, LLMLingua-style.
   */
  async prune(text: string, keep = 0.5): Promise<string> {
    await this.init();
    const tok = this.tokenizer as any;
    const model = this.model as any;

    // Tokenize once; extract ids from the tensor (works across SDK versions).
    const inputs = await tok(text);
    const ids: number[] = Array.from(inputs.input_ids.data as ArrayLike<unknown>, (x) => Number(x));
    if (ids.length <= 4) return text;

    // Forward pass → logits [1, seq, vocab].
    const out = await model(inputs);
    const logits = out.logits;
    const [, seq, vocab] = logits.dims as [number, number, number];
    const data: Float32Array = logits.data;

    // Surprisal of token i+1 = −log softmax(logits[i])[ ids[i+1] ].
    const surprisal = new Float64Array(ids.length);
    surprisal[0] = Infinity; // first token has no context → always keep
    for (let i = 0; i < seq - 1 && i + 1 < ids.length; i++) {
      const base = i * vocab;
      let max = -Infinity;
      for (let v = 0; v < vocab; v++) {
        const x = data[base + v]!;
        if (x > max) max = x;
      }
      let sum = 0;
      for (let v = 0; v < vocab; v++) sum += Math.exp(data[base + v]! - max);
      const logZ = max + Math.log(sum);
      const target = ids[i + 1]!;
      surprisal[i + 1] = logZ - data[base + target]!; // −logprob
    }

    // Keep the top-`keep` fraction by surprisal, preserving order.
    const order = Array.from(surprisal.keys()).sort((a, b) => surprisal[b]! - surprisal[a]!);
    const keepCount = Math.max(1, Math.round(ids.length * Math.min(1, Math.max(0.05, keep))));
    const keepSet = new Set(order.slice(0, keepCount));
    const keptIds = ids.filter((_, i) => keepSet.has(i));

    return tok.decode(keptIds, { skip_special_tokens: true }).trim();
  }
}
