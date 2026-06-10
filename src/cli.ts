#!/usr/bin/env node
/**
 * gist CLI — compress a file (or stdin) and print the result.
 *
 *   gist big.log                 # compress a file, print to stdout
 *   gist big.log --aggressive    # squeeze harder
 *   gist big.log --type=json     # force a compressor
 *   gist big.log | clip          # Windows: copy compressed result to clipboard
 *   cat big.log | gist           # read from stdin
 *
 * Compressed content goes to stdout (pipe-friendly); a one-line stats report
 * goes to stderr. Lets you compress huge content BEFORE pasting it into a chat,
 * so the raw bulk never costs you tokens.
 */
import { readFileSync } from "node:fs";
import { compressContent, type ContentType } from "./compress/router.js";

const args = process.argv.slice(2);
const aggressive = args.includes("--aggressive") || args.includes("-a");
const typeArg = args.find((a) => a.startsWith("--type="))?.split("=")[1];
const file = args.find((a) => !a.startsWith("-"));

if (!file && process.stdin.isTTY) {
  process.stderr.write(
    "usage: gist <file> [--aggressive] [--type=log|json|code|prose|trace|table]\n" +
      "       cat <file> | gist\n"
  );
  process.exit(1);
}

let input: string;
try {
  input = readFileSync(file ?? 0, "utf8"); // fd 0 = stdin
} catch (e) {
  process.stderr.write(`gist: cannot read ${file ?? "stdin"}: ${(e as Error).message}\n`);
  process.exit(1);
}

const r = compressContent(input, {
  level: aggressive ? "aggressive" : "safe",
  type: typeArg as ContentType | undefined,
});

process.stderr.write(
  `gist: ${r.type} · ${r.stats.originalTokens}→${r.stats.compressedTokens} tok (${Math.round(r.stats.saved * 100)}% saved)\n`
);
process.stdout.write(r.text + "\n");
