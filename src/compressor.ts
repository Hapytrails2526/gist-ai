import { compressContent, type CompressOptions, type CompressResult } from "./compress/router.js";
import { ReversibleStore } from "./compress/ccr.js";
import { packContext, type PackOptions } from "./context/packer.js";
import { estimateTokens } from "./context/tokens.js";
import { MemoryStore, type MemorySnapshot } from "./memory/store.js";
import type {
  CompressorOptions,
  Message,
  PackedContext,
  TokenCounter,
} from "./types.js";

/**
 * gist — the one-line facade your apps talk to.
 *
 *   const g = new Compressor({ provider });
 *   await g.ingest(messages);
 *   const ctx = await g.buildContext({ query });
 *   // send ctx.messages to the LLM; read ctx.stats for the savings.
 *
 * Combines Layer 1 (MemoryStore) and Layer 2 (ContextPacker) behind a single
 * surface so the same three lines work in any app.
 */
export class Compressor {
  private readonly store: MemoryStore;
  private readonly budget: number;
  private readonly count: TokenCounter;
  private readonly provider: CompressorOptions["provider"];

  constructor(opts: CompressorOptions) {
    this.provider = opts.provider;
    this.budget = opts.budget ?? 8000;
    this.count = opts.countTokens ?? estimateTokens;
    this.store = new MemoryStore({
      provider: opts.provider,
      keepRecent: opts.keepRecent,
      countTokens: this.count,
    });
  }

  /** Add one or more turns. Distillation happens automatically as needed. */
  async ingest(messages: Message | Message[]): Promise<void> {
    await this.store.ingest(messages);
  }

  /** Assemble a budget-fitted context, optionally ranked by a query. */
  async buildContext(
    opts: { query?: string; budget?: number; system?: string } = {}
  ): Promise<PackedContext> {
    const packOpts: PackOptions = {
      budget: opts.budget ?? this.budget,
      countTokens: this.count,
      query: opts.query,
      provider: this.provider,
      system: opts.system,
      rawTokens: this.store.rawTokensTotal,
    };
    return packContext(this.store.recent, this.store.memories, packOpts);
  }

  /** Export memory as a JSON-serializable snapshot (for persistence). */
  snapshot(): MemorySnapshot {
    return this.store.snapshot();
  }

  /** Restore memory from a snapshot saved earlier. */
  restore(s: Partial<MemorySnapshot>): void {
    this.store.restore(s);
  }

  /** Shared CCR store so compressed content stays retrievable across calls. */
  readonly ccr = new ReversibleStore();

  /**
   * Compress a chunk of tool output (log, JSON, code, or prose) before it goes
   * to the model — Layer-2 content compression, Headroom-style. Routes by type,
   * caches the original in `this.ccr`, and appends a retrieve-handle so nothing
   * is lost. Use `this.ccr.retrieve(handle)` to recover the full original.
   */
  compress(content: string, opts: Omit<CompressOptions, "reversible"> = {}): CompressResult {
    return compressContent(content, { ...opts, reversible: this.ccr });
  }

  /** Inspect current memory state (facts + verbatim backlog). */
  inspect() {
    return {
      memories: this.store.memories,
      recent: this.store.recent,
    };
  }
}
