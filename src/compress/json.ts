/**
 * JSON compressor — collapse repetitive structure, keep anomalies.
 *
 * Big JSON payloads (API responses, search results) are mostly repetition. We
 * minify, truncate long arrays to a representative sample + count, and clip
 * oversized strings — while keeping objects whose shape differs from the array's
 * norm (likely the interesting/anomalous ones, e.g. an item carrying an `error`).
 * Falls back to returning the input unchanged if it isn't valid JSON.
 */
export interface JsonCompressOptions {
  /** Keep the first N items of a long array verbatim. Default 2. */
  sample?: number;
  /** Clip string values longer than this many chars. Default 200. */
  maxString?: number;
}

const ANOMALY = /error|fail|exception|warn|null|missing|invalid/i;

export function compressJson(text: string, opts: JsonCompressOptions = {}): string {
  let data: unknown;
  try {
    data = JSON.parse(text);
  } catch {
    return text;
  }
  const sample = opts.sample ?? 2;
  const maxString = opts.maxString ?? 200;

  const walk = (v: unknown): unknown => {
    if (Array.isArray(v)) {
      if (v.length <= sample + 1) return v.map(walk);
      const head = v.slice(0, sample).map(walk);
      // keep items that look anomalous relative to the bulk
      const anomalies = v
        .slice(sample)
        .filter((it) => ANOMALY.test(JSON.stringify(it)))
        .slice(0, 3)
        .map(walk);
      const omitted = v.length - sample - anomalies.length;
      return [...head, ...anomalies, `…(${omitted} more similar items)`];
    }
    if (v && typeof v === "object") {
      const o: Record<string, unknown> = {};
      for (const k of Object.keys(v as Record<string, unknown>)) {
        o[k] = walk((v as Record<string, unknown>)[k]);
      }
      return o;
    }
    if (typeof v === "string" && v.length > maxString) {
      return `${v.slice(0, maxString)}…(${v.length} chars)`;
    }
    return v;
  };

  return JSON.stringify(walk(data));
}
