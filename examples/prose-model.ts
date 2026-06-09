/**
 * Trained-model prose pruner demo (optional).
 *
 *   npm i @huggingface/transformers   # one-time
 *   npm run demo:prose-model          # first run downloads ~tens of MB
 *
 * Compares the zero-dependency heuristic pruner against the perplexity pruner
 * (a real distilgpt2 via transformers.js), at the same keep ratio.
 */
import { pruneProse, ModelProsePruner, estimateTokens } from "../src/index.js";

const text =
  "The quarterly report indicates that revenue grew by approximately 23 percent " +
  "year over year, which is a really significant improvement, and it was largely " +
  "driven by the new enterprise tier that we launched in March of this year, with " +
  "the deadline for the next release being the end of the month.";

async function main() {
  console.log("=== prose pruning: heuristic vs trained model ===\n");
  console.log(`original (${estimateTokens(text)} tok):\n  ${text}\n`);

  const h = pruneProse(text, { keep: 0.5 });
  console.log(`heuristic   keep=0.5 (${estimateTokens(h)} tok):\n  ${h}\n`);

  const pruner = new ModelProsePruner(); // Xenova/distilgpt2
  console.log("loading distilgpt2 via transformers.js (first run downloads)...");
  const m = await pruner.prune(text, 0.5);
  console.log(`perplexity  keep=0.5 (${estimateTokens(m)} tok):\n  ${m}`);
}

main().catch((e) => {
  console.error("\n(prose-model demo needs '@huggingface/transformers' installed)\n", e?.message ?? e);
  process.exit(1);
});
