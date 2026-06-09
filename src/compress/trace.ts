/**
 * Stack-trace & test-output compressor.
 *
 * The content that floods an agent's context during debugging: deep stack
 * traces (mostly framework/dependency frames) and test runs (walls of passing
 * tests around one failure). Strategy:
 *   - keep the error/failure lines and YOUR-code stack frames
 *   - collapse dependency frames (node_modules, site-packages, node internals)
 *     into "… N library frames …"
 *   - collapse passing tests into "(N passing tests)"
 *   - keep test summaries and assertion diffs
 *
 * Typically 70–90% on real traces/test logs while keeping everything you need
 * to find the bug. `aggressive` additionally drops non-error chatter.
 */
const FRAME = /^\s*(?:at\s|File\s")/;
const LIBFRAME = /node_modules|site-packages|[\\/]dist[\\/]|internal[\\/]|\(node:|<anonymous>|[\\/]usr[\\/]lib|runtime\/|\.pyenv|dist-packages/i;
// \w*Error / \w*Exception catches TypeError:, ReferenceError:, ValueError, etc.
const FAIL_MARK = /✗|✕|✘|×|✖|●|\bFAIL(?:ED)?\b|\bnot ok\b|\w*Error\b|\w*Exception\b|Traceback/;
const PASS_MARK = /^\s*(?:✓|✔|√|PASS\b|ok\s+\d+\b)/;
const SUMMARY = /\b\d+\s+(?:passing|failing|passed|failed|skipped|pending)\b|Tests?:\s|Test Suites?:\s/i;
const CTX = /expected|received|assert|caused by|^\s*[-+>]|=>|diff|line\s+\d+/i;

export function compressTrace(text: string, level: "safe" | "aggressive" = "safe"): string {
  const lines = text.split(/\r?\n/);
  const out: string[] = [];
  let lib = 0;
  let pass = 0;

  const flushLib = (): void => {
    if (lib > 0) {
      out.push(`    … ${lib} library frame${lib === 1 ? "" : "s"} …`);
      lib = 0;
    }
  };
  const flushPass = (): void => {
    if (pass > 0) {
      out.push(`(${pass} passing test${pass === 1 ? "" : "s"})`);
      pass = 0;
    }
  };

  let headerSeen = false;
  for (const l of lines) {
    if (!l.trim()) continue;
    // The first non-blank, non-pass line is the error/header — always keep it.
    if (!headerSeen && !PASS_MARK.test(l) && !FRAME.test(l)) {
      headerSeen = true;
      out.push(l);
      continue;
    }
    headerSeen = true;
    if (FRAME.test(l)) {
      if (LIBFRAME.test(l)) {
        lib++; // dependency/internal frame → collapse
        continue;
      }
      flushLib();
      out.push(l); // your-code frame → keep
      continue;
    }
    flushLib();
    if (FAIL_MARK.test(l)) {
      flushPass();
      out.push(l); // failures and errors → keep verbatim
      continue;
    }
    if (PASS_MARK.test(l)) {
      pass++; // passing test → collapse to a count
      continue;
    }
    if (SUMMARY.test(l)) {
      flushPass();
      out.push(l); // "Tests: 3 failed, 211 passed" → keep
      continue;
    }
    if (level === "safe" || CTX.test(l)) out.push(l);
    // aggressive drops remaining non-error chatter
  }
  flushLib();
  flushPass();
  return out.join("\n");
}
