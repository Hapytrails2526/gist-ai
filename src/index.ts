/**
 * gist — memory & context compression for AI applications.
 * Keeps the gist, drops the bulk.
 */

export { Compressor } from "./compressor.js";
export { MemoryStore, type MemorySnapshot } from "./memory/store.js";
export { distill, dedupe } from "./memory/distiller.js";
export { packContext } from "./context/packer.js";
export { estimateTokens, estimateMany } from "./context/tokens.js";
export {
  ScalarQuantizer,
  BinaryQuantizer,
  cosine,
  type VectorCompressor,
} from "./vector/compressor.js";
export { ProductQuantizer, type PQOptions } from "./vector/pq.js";
export { searchWithRerank, rerankByCosine } from "./vector/search.js";
export { TurboQuantWasm, type TurboQuantOptions } from "./vector/turboquant.js";
export {
  compressContent,
  detectType,
  type ContentType,
  type CompressOptions,
  type CompressResult,
} from "./compress/router.js";
export { ReversibleStore } from "./compress/ccr.js";
export { pruneProse, type PruneOptions } from "./compress/prose.js";
export { ModelProsePruner, type ModelProsePrunerOptions } from "./compress/prose-model.js";
export { compressLog } from "./compress/logs.js";
export { compressJson } from "./compress/json.js";
export { compressCode } from "./compress/code.js";

export { MockProvider } from "./providers/mock.js";
export { ClaudeProvider, type ClaudeProviderOptions } from "./providers/claude.js";

export type {
  Role,
  Message,
  Memory,
  LLMProvider,
  TokenCounter,
  CompressorOptions,
  PackedContext,
  CompressionStats,
} from "./types.js";
