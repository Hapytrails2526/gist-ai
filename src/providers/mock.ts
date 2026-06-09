import type { LLMProvider } from "../types.js";

/**
 * A deterministic, offline provider for tests and the demo.
 *
 * - `complete` does heuristic extractive "distillation": it pulls the most
 *   information-dense sentences out of the prompt's conversation block. No
 *   network, no API key, fully reproducible.
 * - `embed` produces a cheap deterministic hashing embedding so relevance
 *   ranking can be exercised without a real model.
 *
 * Swap this for ClaudeProvider in production — the interface is identical.
 */
export class MockProvider implements LLMProvider {
  async complete(prompt: string): Promise<string> {
    // The distiller passes the raw turns after a "---" marker.
    const body = prompt.includes("---")
      ? prompt.slice(prompt.lastIndexOf("---") + 3)
      : prompt;

    const sentences = body
      .split(/(?<=[.!?])\s+|\n+/)
      .map((s) => s.trim())
      .filter((s) => s.length > 0);

    // Rank by a crude salience signal: length + presence of "signal" words.
    const signal = /\b(want|need|prefer|must|should|decided|use|is|are|named|called|because|so that|goal)\b/i;
    const ranked = sentences
      .map((s) => ({
        s,
        score: (signal.test(s) ? 5 : 0) + Math.min(s.length / 40, 4),
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, Math.max(1, Math.ceil(sentences.length / 4)))
      .map((r) => `- ${r.s.replace(/^(user|assistant)\s*:\s*/i, "")}`);

    return ranked.join("\n");
  }

  async embed(texts: string[]): Promise<number[][]> {
    const DIM = 64;
    return texts.map((t) => {
      const v = new Array<number>(DIM).fill(0);
      for (const tok of t.toLowerCase().split(/\W+/)) {
        if (!tok) continue;
        let h = 2166136261;
        for (let i = 0; i < tok.length; i++) {
          h ^= tok.charCodeAt(i);
          h = Math.imul(h, 16777619);
        }
        const idx = Math.abs(h) % DIM;
        v[idx] = (v[idx] ?? 0) + 1;
      }
      // L2 normalize so cosine similarity is well-behaved.
      const norm = Math.hypot(...v) || 1;
      return v.map((x) => x / norm);
    });
  }
}
