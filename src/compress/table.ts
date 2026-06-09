/**
 * CSV / tabular compressor.
 *
 * Query results, exported tables, and markdown tables are mostly rows that look
 * alike. Keep the header (column names), a few representative rows, and any rows
 * that look anomalous (error/null/empty cells); summarize the rest as a count.
 * On a 1000-row result that's ~99% smaller while the agent still sees the shape
 * of the data and anything unusual.
 *
 * Handles comma, tab, and pipe (markdown) delimiters; markdown separator rows
 * (|---|---|) are dropped.
 */
const ANOMALY = /\b(?:error|null|nan|none|missing|undefined|fail(?:ed)?|invalid|timeout)\b|,,|\|\s*\||^\s*,|,\s*$/i;
const SEPARATOR = /^[\s|:.-]*-[\s|:.-]*$/; // markdown header rule, e.g. |---|:--:|

/** Detect a consistent delimiter (and column count) — also used for routing. */
export function pickDelimiter(lines: string[]): { delim: string; cols: number } | null {
  for (const delim of ["\t", "|", ","]) {
    const counts = lines.filter((l) => !SEPARATOR.test(l)).map((l) => l.split(delim).length - 1);
    const positive = counts.filter((c) => c > 0);
    if (positive.length < Math.max(2, lines.length * 0.6)) continue;
    const freq = new Map<number, number>();
    for (const c of positive) freq.set(c, (freq.get(c) ?? 0) + 1);
    let mode = 0;
    let modeN = 0;
    for (const [c, n] of freq) if (n > modeN) ((mode = c), (modeN = n));
    if (modeN >= positive.length * 0.6 && mode >= 1) return { delim, cols: mode + 1 };
  }
  return null;
}

export function compressTable(text: string, level: "safe" | "aggressive" = "safe"): string {
  const all = text.split(/\r?\n/).filter((l) => l.trim());
  const picked = pickDelimiter(all);
  if (!picked) return text;

  const rows = all.filter((l) => !SEPARATOR.test(l)); // drop markdown rule rows
  if (rows.length < 3) return text;

  const header = rows[0]!;
  const data = rows.slice(1);
  const sampleN = level === "aggressive" ? 1 : 3;
  const head = data.slice(0, sampleN);
  const anomalies = data.slice(sampleN).filter((l) => ANOMALY.test(l)).slice(0, 3);
  const omitted = data.length - head.length - anomalies.length;

  const out = [header, ...head, ...anomalies];
  if (omitted > 0) out.push(`… ${omitted} more rows …`);
  return out.join("\n");
}
