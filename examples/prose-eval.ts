/**
 * Prose pruner head-to-head: heuristic vs trained (perplexity) model.
 *
 *   npm i @huggingface/transformers   # one-time
 *   npm run eval:prose
 *
 * At matched keep-ratios, measures saved% AND signal-recall (must-survive
 * markers) for both pruners. The trained model "wins" only if it preserves the
 * markers at a HARDER ratio than the heuristic — otherwise the heuristic's
 * zero-dependency speed makes it the better default.
 */
import { pruneProse, ModelProsePruner, estimateTokens } from "../src/index.js";

interface Sample {
  text: string;
  mustSurvive: string[];
}

const samples: Sample[] = [
  {
    text:
      "The quarterly report indicates that revenue grew by approximately 23 percent year over year, " +
      "which is a really significant improvement, and it was largely driven by the new enterprise tier " +
      "that we launched in March of this year, with the deadline for the next release being the end of the month.",
    mustSurvive: ["23", "enterprise", "March"],
  },
  {
    text:
      "During the incident, the on-call engineer Maria discovered that the payment service was timing out " +
      "because the connection pool to the Postgres database had been exhausted, so she increased the pool " +
      "size from 20 to 80 and the error rate dropped back to normal within about five minutes.",
    mustSurvive: ["Maria", "Postgres", "pool", "80"],
  },
  {
    text:
      "Our recommendation is to migrate the analytics pipeline from Redshift to BigQuery before the fourth " +
      "quarter, primarily because the team estimated a roughly forty percent reduction in monthly cost and " +
      "a meaningful improvement in query latency for the larger dashboards used by the finance department.",
    mustSurvive: ["Redshift", "BigQuery", "forty percent", "finance"],
  },
];

const KEEPS = [0.5, 0.4, 0.3];

async function main() {
  const model = new ModelProsePruner(); // Xenova/distilgpt2
  console.log("=== prose pruning: heuristic vs trained (perplexity) ===\n");
  console.log("loading distilgpt2 (first run downloads)…\n");
  // warm up
  await model.prune("warmup text to load the model", 0.5);

  for (let i = 0; i < samples.length; i++) {
    const s = samples[i]!;
    const inTok = estimateTokens(s.text);
    console.log(`sample ${i + 1} (${inTok} tok, markers: ${s.mustSurvive.join(", ")})`);
    console.log("  keep   heuristic saved/recall   trained saved/recall");
    for (const keep of KEEPS) {
      const h = pruneProse(s.text, { keep });
      const m = await model.prune(s.text, keep);
      const hSaved = Math.round((1 - estimateTokens(h) / inTok) * 100);
      const mSaved = Math.round((1 - estimateTokens(m) / inTok) * 100);
      const hRec = s.mustSurvive.filter((x) => h.includes(x)).length;
      const mRec = s.mustSurvive.filter((x) => m.includes(x)).length;
      const n = s.mustSurvive.length;
      console.log(
        `  ${keep.toFixed(2)}      ${String(hSaved).padStart(3)}%   ${hRec}/${n}            ${String(mSaved).padStart(3)}%   ${mRec}/${n}`
      );
    }
    console.log();
  }
  console.log("Win = preserves all markers at a higher saved% than the other.");
}

main().catch((e) => {
  console.error("\n(needs '@huggingface/transformers' installed)\n", e?.message ?? e);
  process.exit(1);
});
