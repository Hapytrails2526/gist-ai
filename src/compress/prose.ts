/**
 * Prose token pruner — LLMLingua-2-INSPIRED.
 *
 * LLMLingua-2 trains a ModernBERT token classifier to keep high-information
 * tokens and drop the rest (task-agnostic, ~30–50% reduction). We can't ship a
 * trained model in a dependency-free TS library, so this is a fast HEURISTIC
 * approximation of the same idea: score each token by information signal
 * (entities, numbers, identifiers, rarity, length; penalize stopwords and
 * redundancy), then keep the top fraction in original order. The result is
 * information-dense, not fluent — exactly like LLMLingua output.
 *
 * For higher fidelity, pair with a provider: ask the LLM to compress. This
 * heuristic is the zero-dependency default.
 */

// Common low-information English words.
const STOPWORDS = new Set(
  "a an the of to in on at for and or but nor so yet is are was were be been being am it its this that these those with as by from into onto than then there here i you he she they we us them him her my your his their our me do does did has have had will would shall can could should may might must not no yes just very really quite too also about over under again more most some any all each every".split(
    /\s+/
  )
);

export interface PruneOptions {
  /** Fraction of tokens to KEEP (0..1). Default 0.5 (≈2× compression). */
  keep?: number;
}

const normalize = (t: string): string => t.toLowerCase().replace(/[^a-z0-9]/g, "");

/** Drop low-salience tokens, keeping ~`keep` of them in original order. */
export function pruneProse(text: string, opts: PruneOptions = {}): string {
  const keep = Math.min(1, Math.max(0.05, opts.keep ?? 0.5));
  const tokens = text.match(/\S+/g) ?? [];
  if (tokens.length <= 3) return text;

  const freq = new Map<string, number>();
  for (const t of tokens) {
    const n = normalize(t);
    if (n) freq.set(n, (freq.get(n) ?? 0) + 1);
  }
  let maxFreq = 1;
  for (const v of freq.values()) if (v > maxFreq) maxFreq = v;

  const score = (t: string): number => {
    const n = normalize(t);
    if (!n) return 0.05; // pure punctuation
    let s = 1;
    if (STOPWORDS.has(n)) s -= 0.85;
    if (/\d/.test(t)) s += 1.5; // numbers carry info
    if (/^[A-Z]/.test(t)) s += 0.6; // capitalized → entities/proper nouns
    if (/[_/.\\{}()=<>:#@[\]]/.test(t)) s += 0.8; // identifiers / code-ish
    s += Math.min(n.length / 8, 0.8); // longer → more specific
    s -= 0.4 * ((freq.get(n) ?? 1) / maxFreq); // redundant → less useful
    return s;
  };

  const scored = tokens.map((t, i) => ({ i, s: score(t) }));
  const keepCount = Math.max(1, Math.round(tokens.length * keep));
  const keepIdx = new Set(
    scored
      .slice()
      .sort((a, b) => b.s - a.s)
      .slice(0, keepCount)
      .map((x) => x.i)
  );
  return tokens.filter((_, i) => keepIdx.has(i)).join(" ");
}
