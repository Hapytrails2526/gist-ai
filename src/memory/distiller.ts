import type { LLMProvider, Memory, Message } from "../types.js";

let counter = 0;
function nextId(): string {
  counter += 1;
  return `m${counter.toString(36)}`;
}

const DISTILL_PROMPT = `You compress conversation into durable memory.
Extract only facts that will still matter in future turns: user preferences,
decisions made, named entities, goals, and open tasks. One idea per line,
prefixed with "- ". Drop pleasantries, redundancy, and anything ephemeral.
Be terse. Conversation follows the marker.
---`;

/**
 * Distill a block of messages into compact, durable Memory facts.
 *
 * Layer 1 of gist: this is where bulk conversation collapses into the "gist".
 */
export async function distill(
  messages: Message[],
  provider: LLMProvider,
  range: { from: number; to: number }
): Promise<Memory[]> {
  if (messages.length === 0) return [];

  const transcript = messages
    .map((m) => `${m.role}: ${m.content}`)
    .join("\n");

  const out = await provider.complete(`${DISTILL_PROMPT}\n${transcript}`, {
    maxTokens: 512,
  });

  return out
    .split("\n")
    .map((l) => l.replace(/^[-*]\s*/, "").trim())
    .filter((l) => l.length > 0)
    .map((text) => ({ id: nextId(), text, source: range, kind: "fact" as const }));
}

/**
 * Drop near-duplicate memories. Cheap lexical Jaccard over word sets — good
 * enough to stop the same fact accumulating across many distillation passes.
 */
export function dedupe(memories: Memory[], threshold = 0.8): Memory[] {
  const kept: Memory[] = [];
  const sets = new Map<string, Set<string>>();

  const words = (t: string) =>
    new Set(t.toLowerCase().split(/\W+/).filter(Boolean));

  for (const m of memories) {
    const ws = words(m.text);
    let dup = false;
    for (const k of kept) {
      const ks = sets.get(k.id)!;
      const inter = [...ws].filter((w) => ks.has(w)).length;
      const union = new Set([...ws, ...ks]).size || 1;
      if (inter / union >= threshold) {
        dup = true;
        break;
      }
    }
    if (!dup) {
      kept.push(m);
      sets.set(m.id, ws);
    }
  }
  return kept;
}
