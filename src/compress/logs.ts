/**
 * Log compressor — keep the signal, drop the noise.
 *
 * The "10,000-token log to find one error line" case. Strategy: always keep
 * error/warning/stack lines; dedupe routine lines by template (numbers and
 * timestamps normalized) so repeats collapse; drop blanks. Routine lines after
 * their first occurrence are summarized as a count. Typically 80–95% on noisy
 * build/test/SRE logs while preserving every failure.
 */
const KEEP = /\b(error|err|fatal|critical|crit|exception|traceback|panic|fail(?:ed|ure)?|warn(?:ing)?|assert(?:ion)?)\b/i;
const STACK = /^\s+at\s|^\s+File\s"|^Traceback|^\s+\.\.\./;

const templ = (l: string): string =>
  l
    .replace(/\d{4}-\d{2}-\d{2}[ T][\d:.,]+Z?/g, "<ts>")
    .replace(/\b\d+(?:\.\d+)?\b/g, "<n>")
    .replace(/0x[0-9a-fA-F]+/g, "<hex>")
    .replace(/\b[0-9a-f]{8,}\b/gi, "<hash>");

interface Group {
  first: string; // full first line (used when the template occurs once)
  rep: string; // timestamp-stripped, number-generalized template
  count: number;
}

export function compressLog(text: string): string {
  const lines = text.split(/\r?\n/);
  const out: Array<string | Group> = [];
  const groups = new Map<string, Group>();

  for (const l of lines) {
    if (KEEP.test(l) || STACK.test(l)) {
      out.push(l); // always keep failures and stack frames, verbatim
      continue;
    }
    if (!l.trim()) continue; // drop blanks
    const t = templ(l);
    let g = groups.get(t);
    if (!g) {
      const rep = l
        .replace(/^\s*\d{4}-\d{2}-\d{2}[ T][\d:.,]+Z?\s*/, "") // drop leading timestamp
        .replace(/\b\d+(?:\.\d+)?\b/g, "#"); // generalize varying numbers
      g = { first: l, rep, count: 0 };
      groups.set(t, g);
      out.push(g);
    }
    g.count++;
  }

  // Repeated routine lines collapse to "N× <template>" (timestamps + values
  // dropped); a line that occurred once is kept in full.
  return out
    .map((item) =>
      typeof item === "string" ? item : item.count > 1 ? `${item.count}× ${item.rep}` : item.first
    )
    .join("\n");
}
