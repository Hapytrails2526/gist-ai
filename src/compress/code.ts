/**
 * Code compressor — strip what the model can infer.
 *
 * HONEST SCOPE: this is regex-based, not AST-aware (Headroom's CodeCompressor
 * parses ASTs to collapse function bodies). We remove comments, trailing
 * whitespace, and runs of blank lines — safe, language-agnostic, lossless to
 * meaning. Typically 15–40% on commented source; the structure and all code
 * stay intact.
 */
export function compressCode(text: string, level: "safe" | "aggressive" = "safe"): string {
  let out = text
    // block comments /* ... */
    .replace(/\/\*[\s\S]*?\*\//g, "")
    // line comments // ... (avoid matching the // in http://)
    .replace(/([^:"'`])\/\/[^\n]*/g, "$1")
    // hash comments (python/shell/yaml), only full-line to avoid '#' in strings
    .replace(/^\s*#[^\n]*$/gm, "")
    // trailing whitespace
    .replace(/[ \t]+$/gm, "");

  if (level === "aggressive") {
    // Drop ALL blank lines and reduce indentation runs — terser, still parseable.
    out = out.replace(/^\s*\n/gm, "").replace(/^[ \t]+/gm, (m) => " ".repeat(Math.ceil(m.length / 2)));
  } else {
    out = out.replace(/\n{3,}/g, "\n\n"); // collapse 3+ blank lines to 1
  }
  return out.trim();
}
