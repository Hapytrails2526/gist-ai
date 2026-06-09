/**
 * Runnable demo — no API key needed (uses the offline MockProvider).
 *
 *   npm install
 *   npm run demo
 *
 * It builds a long fake conversation, feeds it through gist, then prints the
 * raw-vs-packed token counts so you can see the compression ratio directly.
 */
import { Compressor, MockProvider } from "../src/index.js";
import { ScalarQuantizer, BinaryQuantizer } from "../src/index.js";

function fakeConversation(turns: number) {
  const topics = [
    "I'm building a Tauri video editor called CutMind PRO.",
    "It needs Riverside-style multi-track recording.",
    "Use Rust for the backend, the heavy export pipeline runs there.",
    "I prefer dark UI themes, and keyboard-first shortcuts.",
    "The export must support 4K at 60fps, that's a hard requirement.",
    "Remember the project lives in Downloads\\CutMindPRO.",
    "Can you also reorder the timeline clips by drag and drop?",
    "By the way, my deadline for the beta is the end of the month.",
  ];
  const msgs = [];
  for (let i = 0; i < turns; i++) {
    const topic = topics[i % topics.length]!;
    msgs.push({ role: "user" as const, content: topic });
    msgs.push({
      role: "assistant" as const,
      content: `Got it. ${topic} Here's a long, chatty acknowledgement with lots of filler words that pad the token count but carry little durable signal, restating what you said and adding pleasantries before moving on to the next point.`,
    });
  }
  return msgs;
}

async function main() {
  const provider = new MockProvider();
  const gist = new Compressor({ provider, budget: 400, keepRecent: 4 });

  const convo = fakeConversation(12); // 24 messages
  for (const m of convo) await gist.ingest(m);

  const ctx = await gist.buildContext({
    query: "What are the hard requirements for the CutMind export pipeline?",
    system: "You are a coding assistant for the CutMind PRO project.",
  });

  console.log("=== gist compression demo ===\n");
  console.log(`raw messages:     ${ctx.stats.rawMessages}`);
  console.log(`packed messages:  ${ctx.stats.packedMessages}`);
  console.log(`memories distilled: ${gist.inspect().memories.length}`);
  console.log(`memories used:    ${ctx.stats.memoriesUsed}\n`);
  console.log(`raw tokens:       ${ctx.stats.rawTokens}`);
  console.log(`packed tokens:    ${ctx.stats.packedTokens}`);
  console.log(`compression:      ${ctx.stats.ratio}x smaller\n`);

  console.log("--- distilled memory ---");
  for (const m of gist.inspect().memories) console.log(`  • ${m.text}`);

  console.log("\n--- packed context sent to LLM ---");
  for (const m of ctx.messages) {
    const preview = m.content.length > 80 ? m.content.slice(0, 77) + "..." : m.content;
    console.log(`  [${m.role}] ${preview}`);
  }

  // Layer 3 preview: embedding compression ratios.
  console.log("\n--- Layer 3: embedding compression (per 1536-dim vector) ---");
  const dim = 1536;
  const float32 = dim * 4;
  for (const q of [new ScalarQuantizer(), new BinaryQuantizer()]) {
    const bytes = q.bytesPer(dim);
    console.log(
      `  ${q.name.padEnd(12)} ${bytes} bytes  (${(float32 / bytes).toFixed(0)}x smaller than float32)`
    );
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
