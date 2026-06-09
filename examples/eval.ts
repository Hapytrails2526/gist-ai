/**
 * Compression eval harness.
 *
 *   npm run eval
 *
 * Scores each compressor on TWO axes:
 *   - saved%       : how many tokens removed
 *   - signal-recall: fraction of "must-survive" markers still present
 *
 * The goal is to push saved% up while keeping signal-recall at 100% — dropping
 * an error or key value is a failure, no matter how good the ratio looks.
 */
import { compressContent, estimateTokens } from "../src/index.js";

interface Sample {
  name: string;
  type?: "log" | "json" | "code" | "prose" | "trace" | "table";
  text: string;
  mustSurvive: string[];
}

const csv = [
  "id,name,status,latency_ms",
  ...Array.from({ length: 40 }, (_, i) => `${i + 1},svc-${i + 1},ok,${10 + (i % 7)}`),
  "41,svc-billing,error,timeout",
  ...Array.from({ length: 18 }, (_, i) => `${42 + i},svc-${42 + i},ok,${12 + (i % 5)}`),
].join("\n");

const mdTable = [
  "| region | users | revenue |",
  "|--------|-------|---------|",
  "| US     | 1200  | 45000   |",
  "| EU     | 900   | 32000   |",
  "| LATAM  | 400   | 11000   |",
  "| MEA    | 300   | 9000    |",
  "| APAC   | 50    | error   |",
].join("\n");

const stackTrace = [
  "TypeError: Cannot read properties of undefined (reading 'id')",
  "    at getUser (src/services/user.js:42:18)",
  "    at handler (src/routes/api.js:88:25)",
  "    at Layer.handle (node_modules/express/lib/router/layer.js:95:5)",
  "    at next (node_modules/express/lib/router/route.js:144:13)",
  "    at Route.dispatch (node_modules/express/lib/router/route.js:114:3)",
  "    at node_modules/express/lib/router/index.js:284:15",
  "    at processTicksAndRejections (node:internal/process/task_queues:95:5)",
].join("\n");

const testOutput = [
  ...Array.from({ length: 18 }, (_, i) => `PASS src/unit/module${i}.test.js`),
  "FAIL src/checkout.test.js",
  "  ● checkout › applies the discount code",
  "    expect(received).toBe(expected)",
  "    Expected: 90",
  "    Received: 100",
  "      at Object.<anonymous> (src/checkout.test.js:31:23)",
  "Tests: 1 failed, 18 passed, 19 total",
].join("\n");

const repetitiveLog = [
  ...Array.from({ length: 60 }, (_, i) => `2026-06-09T10:00:${String(i % 60).padStart(2, "0")}Z INFO worker: job ${i} ok (${10 + (i % 5)}ms)`),
  "2026-06-09T10:01:00Z ERROR worker: job 60 FAILED — OutOfMemory: heap limit 2048MB exceeded",
  ...Array.from({ length: 40 }, (_, i) => `2026-06-09T10:01:${String(i % 60).padStart(2, "0")}Z INFO worker: job ${61 + i} ok (${10 + (i % 5)}ms)`),
].join("\n");

const variedLog = [
  ...Array.from({ length: 10 }, () => `2026-06-09T14:02:01Z INFO api: GET /health 200 3ms`),
  "2026-06-09T14:02:12Z ERROR api: POST /payments 500 2200ms — Stripe timeout (idempotency_key=pay_8812)",
  "2026-06-09T14:02:12Z ERROR api:   at chargeCard (billing/stripe.js:88:14)",
  "2026-06-09T14:02:19Z WARN api: cache miss rate 38% (threshold 30%)",
  ...Array.from({ length: 6 }, (_, i) => `2026-06-09T14:02:${20 + i}Z INFO api: GET /users/${440 + i} 200 1${i}ms`),
].join("\n");

const json = JSON.stringify({
  results: [
    ...Array.from({ length: 50 }, (_, i) => ({ id: i, name: `user_${i}`, status: "active", score: 0.9 })),
    { id: 999, name: "user_999", status: "error", error: "quota exceeded" },
  ],
});

const code = `// Fibonacci with memoization.
function fib(n) {
  const memo = {}; // cache


  function go(k) {
    if (k < 2) return k;
    if (memo[k]) return memo[k];
    memo[k] = go(k - 1) + go(k - 2);
    return memo[k];
  }
  return go(n);
}`;

const prose =
  "The quarterly report indicates revenue grew approximately 23 percent year over year, " +
  "driven by the new enterprise tier launched in March, with the next release deadline at month end.";

const samples: Sample[] = [
  { name: "log: repetitive (101 lines)", type: "log", text: repetitiveLog, mustSurvive: ["OutOfMemory", "heap limit 2048MB"] },
  { name: "log: varied (19 lines)", type: "log", text: variedLog, mustSurvive: ["Stripe timeout", "pay_8812", "chargeCard", "cache miss rate 38%"] },
  { name: "json: 50 + 1 anomaly", type: "json", text: json, mustSurvive: ["quota exceeded", "error"] },
  { name: "code: commented", type: "code", text: code, mustSurvive: ["function fib", "memo[k] = go(k - 1)"] },
  { name: "prose: report", type: "prose", text: prose, mustSurvive: ["23", "enterprise", "March"] },
  { name: "trace: JS stack", text: stackTrace, mustSurvive: ["Cannot read properties of undefined", "getUser (src/services/user.js:42", "src/routes/api.js:88"] },
  { name: "trace: jest (18 pass, 1 fail)", text: testOutput, mustSurvive: ["FAIL src/checkout.test.js", "applies the discount", "Expected: 90", "1 failed"] },
  { name: "table: CSV (59 rows)", text: csv, mustSurvive: ["id,name,status,latency_ms", "svc-billing", "error"] },
  { name: "table: markdown", text: mdTable, mustSurvive: ["region", "APAC", "error"] },
];

console.log("=== gist compression eval (safe vs aggressive) ===\n");
console.log("sample                          safe%  recall   aggr%  recall");
console.log("------------------------------  -----  ------   -----  ------");

const tot = { safeIn: 0, safeOut: 0, aggrIn: 0, aggrOut: 0, markers: 0, safeKept: 0, aggrKept: 0 };
for (const s of samples) {
  const safe = compressContent(s.text, { type: s.type, level: "safe" });
  const aggr = compressContent(s.text, { type: s.type, level: "aggressive" });
  const safeKept = s.mustSurvive.filter((m) => safe.text.includes(m)).length;
  const aggrKept = s.mustSurvive.filter((m) => aggr.text.includes(m)).length;
  tot.safeIn += safe.stats.originalTokens;
  tot.safeOut += safe.stats.compressedTokens;
  tot.aggrIn += aggr.stats.originalTokens;
  tot.aggrOut += aggr.stats.compressedTokens;
  tot.markers += s.mustSurvive.length;
  tot.safeKept += safeKept;
  tot.aggrKept += aggrKept;
  const m = s.mustSurvive.length;
  const flag = aggrKept < m ? " ⚠️" : "";
  console.log(
    `${s.name.padEnd(30)}  ${(safe.stats.saved * 100).toFixed(0).padStart(4)}%  ${`${safeKept}/${m}`.padStart(5)}   ${(aggr.stats.saved * 100).toFixed(0).padStart(4)}%  ${`${aggrKept}/${m}`.padStart(5)}${flag}`
  );
}
console.log("------------------------------  -----  ------   -----  ------");
console.log(
  `${"AGGREGATE".padEnd(30)}  ${((1 - tot.safeOut / tot.safeIn) * 100).toFixed(0).padStart(4)}%  ${`${tot.safeKept}/${tot.markers}`.padStart(5)}   ${((1 - tot.aggrOut / tot.aggrIn) * 100).toFixed(0).padStart(4)}%  ${`${tot.aggrKept}/${tot.markers}`.padStart(5)}`
);
console.log(`\nsafe: ${tot.safeIn}→${tot.safeOut} tok · aggressive: ${tot.aggrIn}→${tot.aggrOut} tok (signal must stay ${tot.markers}/${tot.markers})`);
void estimateTokens;
