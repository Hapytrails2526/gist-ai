import { estimateTokens } from "../context/tokens.js";
import { ReversibleStore } from "./ccr.js";
import { pruneProse } from "./prose.js";
import { compressLog } from "./logs.js";
import { compressJson } from "./json.js";
import { compressCode } from "./code.js";

export type ContentType = "json" | "log" | "code" | "prose";

/** Sniff the content type so the right specialist compressor is used. */
export function detectType(text: string): ContentType {
  const t = text.trim();
  if (
    (t.startsWith("{") && t.endsWith("}")) ||
    (t.startsWith("[") && t.endsWith("]"))
  ) {
    try {
      JSON.parse(t);
      return "json";
    } catch {
      /* not JSON, keep sniffing */
    }
  }
  const lines = t.split(/\r?\n/);
  if (lines.length >= 4) {
    const logHits = lines.filter(
      (l) =>
        /\b(error|warn|warning|info|debug|trace|fatal|critical)\b/i.test(l) ||
        /\d{4}-\d{2}-\d{2}[ T]\d/.test(l) ||
        /^\s+at\s/.test(l) ||
        /^\[\d/.test(l)
    ).length;
    if (logHits / lines.length >= 0.3) return "log";
  }
  const codeHits = (t.match(/[{}();]|=>|\b(function|const|let|def|class|import|export|return|public|private|fn|func)\b/g) ?? []).length;
  if (codeHits >= Math.max(8, t.length / 80)) return "code";
  return "prose";
}

export interface CompressOptions {
  /** Force a content type instead of auto-detecting. */
  type?: ContentType;
  /** Prose keep-ratio (passed to the token pruner). */
  keep?: number;
  /** If provided, the original is cached here and a retrieve-handle appended. */
  reversible?: ReversibleStore;
}

export interface CompressResult {
  type: ContentType;
  text: string;
  handle?: string;
  stats: { originalTokens: number; compressedTokens: number; ratio: number; saved: number };
}

/**
 * Route content to the right specialist compressor (Headroom-style), with
 * optional CCR reversibility. Returns the compressed text + a token report.
 */
export function compressContent(input: string, opts: CompressOptions = {}): CompressResult {
  const type = opts.type ?? detectType(input);
  let text: string;
  switch (type) {
    case "json":
      text = compressJson(input);
      break;
    case "log":
      text = compressLog(input);
      break;
    case "code":
      text = compressCode(input);
      break;
    default:
      text = pruneProse(input, { keep: opts.keep });
      break;
  }

  let handle: string | undefined;
  if (opts.reversible) {
    handle = opts.reversible.put(input);
    text = `${text}\n«gist:retrieve ${handle}»`;
  }

  const originalTokens = estimateTokens(input);
  const compressedTokens = estimateTokens(text);
  const ratio = compressedTokens > 0 ? +(originalTokens / compressedTokens).toFixed(2) : 1;
  const saved = originalTokens > 0 ? +(1 - compressedTokens / originalTokens).toFixed(3) : 0;
  return { type, text, handle, stats: { originalTokens, compressedTokens, ratio, saved } };
}
