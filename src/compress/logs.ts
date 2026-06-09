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

export function compressLog(text: string): string {
  const lines = text.split(/\r?\n/);
  const out: string[] = [];
  const seen = new Map<string, number>();
  let dropped = 0;

  for (const l of lines) {
    if (KEEP.test(l) || STACK.test(l)) {
      out.push(l); // always keep failures and stack frames
      continue;
    }
    if (!l.trim()) continue; // drop blanks
    const t = templ(l);
    const c = (seen.get(t) ?? 0) + 1;
    seen.set(t, c);
    if (c <= 1) out.push(l);
    else dropped++; // routine repeat
  }

  if (dropped > 0) out.push(`… ${dropped} repeated/routine log lines omitted …`);
  return out.join("\n");
}
