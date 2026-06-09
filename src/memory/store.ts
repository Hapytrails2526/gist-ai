import type { LLMProvider, Memory, Message, TokenCounter } from "../types.js";
import { estimateTokens } from "../context/tokens.js";
import { dedupe, distill } from "./distiller.js";

export interface MemoryStoreOptions {
  provider: LLMProvider;
  /** Recent turns kept verbatim and never distilled away. */
  keepRecent?: number;
  /** Distill once the verbatim backlog exceeds this many messages. */
  distillEvery?: number;
  /** Token counter, used to track the true uncompressed baseline. */
  countTokens?: TokenCounter;
}

/** JSON-serializable snapshot of a store, for persistence across sessions. */
export interface MemorySnapshot {
  recent: Message[];
  memories: Memory[];
  rawTokensTotal: number;
  ingested: number;
}

/**
 * Tiered conversation memory (Layer 1).
 *
 *   working memory  — the most recent `keepRecent` turns, kept verbatim
 *   long-term memory — older turns, distilled to deduped Memory facts
 *
 * `ingest` appends raw turns; when the verbatim backlog grows past the
 * threshold, the overflow is distilled and folded into long-term memory.
 */
export class MemoryStore {
  private readonly provider: LLMProvider;
  private readonly keepRecent: number;
  private readonly distillEvery: number;
  private readonly count: TokenCounter;

  /** Verbatim, most-recent-last. */
  recent: Message[] = [];
  /** Distilled durable facts. */
  memories: Memory[] = [];
  /** Running count of all messages ever ingested (for source ranges). */
  private ingested = 0;
  /**
   * Total tokens of EVERY turn ever ingested — the honest baseline for what
   * you would have spent dumping the whole transcript. Distillation discards
   * the verbatim originals, so we accumulate this at ingest time.
   */
  rawTokensTotal = 0;

  constructor(opts: MemoryStoreOptions) {
    this.provider = opts.provider;
    this.keepRecent = opts.keepRecent ?? 6;
    this.distillEvery = opts.distillEvery ?? 8;
    this.count = opts.countTokens ?? estimateTokens;
  }

  async ingest(messages: Message | Message[]): Promise<void> {
    const batch = Array.isArray(messages) ? messages : [messages];
    for (const m of batch) {
      this.recent.push(m);
      this.ingested += 1;
      this.rawTokensTotal += this.count(m.content);
    }
    if (this.recent.length > this.keepRecent + this.distillEvery) {
      await this.flush();
    }
  }

  /** Export current state as a JSON-serializable snapshot. */
  snapshot(): MemorySnapshot {
    return {
      recent: this.recent,
      memories: this.memories,
      rawTokensTotal: this.rawTokensTotal,
      ingested: this.ingested,
    };
  }

  /** Restore from a snapshot (e.g. loaded from disk/localStorage). */
  restore(s: Partial<MemorySnapshot>): void {
    this.recent = s.recent ?? [];
    this.memories = s.memories ?? [];
    this.rawTokensTotal = s.rawTokensTotal ?? 0;
    this.ingested = s.ingested ?? this.recent.length;
  }

  /** Force-distill everything except the most recent `keepRecent` turns. */
  async flush(): Promise<void> {
    const overflow = this.recent.length - this.keepRecent;
    if (overflow <= 0) return;

    const toDistill = this.recent.slice(0, overflow);
    this.recent = this.recent.slice(overflow);

    const from = this.ingested - this.recent.length - toDistill.length;
    const fresh = await distill(toDistill, this.provider, {
      from,
      to: from + toDistill.length,
    });
    this.memories = dedupe([...this.memories, ...fresh]);
  }
}
