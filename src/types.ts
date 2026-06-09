/** Core shared types for gist. */

export type Role = "system" | "user" | "assistant";

export interface Message {
  role: Role;
  content: string;
  /** Optional epoch ms; used for recency ordering. Falls back to insertion order. */
  ts?: number;
}

/** A distilled, durable fact extracted from conversation history. */
export interface Memory {
  id: string;
  /** The compressed fact, one idea per memory. */
  text: string;
  /** Optional category to help packing decide what to include. */
  kind?: "fact" | "preference" | "decision" | "entity" | "task";
  /** Source message index range this was distilled from, for traceability. */
  source?: { from: number; to: number };
  /** Optional embedding for relevance ranking (may be quantized). */
  embedding?: number[] | Int8Array | Uint8Array;
  /** Salience score 0..1 used to break ties when over budget. */
  weight?: number;
}

/** A provider abstraction so gist is not tied to any single LLM vendor. */
export interface LLMProvider {
  /** Complete a single prompt and return text. */
  complete(prompt: string, opts?: { maxTokens?: number }): Promise<string>;
  /** Optional: embed text into a vector for relevance ranking. */
  embed?(texts: string[]): Promise<number[][]>;
}

/** Pluggable token counter. Defaults to a fast char-based estimate. */
export type TokenCounter = (text: string) => number;

export interface CompressorOptions {
  provider: LLMProvider;
  /** Total token budget for the assembled context. */
  budget?: number;
  /** How many of the most recent turns to always keep verbatim. */
  keepRecent?: number;
  /** Token counter override (e.g. a real tokenizer). */
  countTokens?: TokenCounter;
}

export interface PackedContext {
  /** Messages ready to send to the LLM, already within budget. */
  messages: Message[];
  /** Compression report. */
  stats: CompressionStats;
}

export interface CompressionStats {
  /** Tokens the raw, uncompressed history would have cost. */
  rawTokens: number;
  /** Tokens the packed context actually costs. */
  packedTokens: number;
  /** rawTokens / packedTokens, e.g. 6.2 means 6.2x smaller. */
  ratio: number;
  /** Counts for transparency. */
  rawMessages: number;
  packedMessages: number;
  memoriesUsed: number;
}
