/**
 * CCR — Compress-Cache-Retrieve (reversible compression, à la Headroom).
 *
 * Aggressive compression is safe when nothing is permanently lost: the original
 * is kept locally and the model is handed a short handle it can "retrieve" on
 * demand. This store holds the originals; the router embeds the handle in the
 * compressed output as a «gist:retrieve <id>» marker.
 */
let counter = 0;

export class ReversibleStore {
  private readonly map = new Map<string, string>();

  /** Stash an original, return a short handle. */
  put(original: string): string {
    const handle = `ccr_${(counter++).toString(36)}`;
    this.map.set(handle, original);
    return handle;
  }

  /** Recover the original for a handle (undefined if unknown/evicted). */
  retrieve(handle: string): string | undefined {
    return this.map.get(handle);
  }

  /** Pull every handle referenced in a compressed string back to originals. */
  retrieveAll(compressed: string): Record<string, string> {
    const out: Record<string, string> = {};
    for (const m of compressed.matchAll(/«gist:retrieve (ccr_[a-z0-9]+)»/g)) {
      const h = m[1]!;
      const v = this.map.get(h);
      if (v !== undefined) out[h] = v;
    }
    return out;
  }

  get size(): number {
    return this.map.size;
  }

  clear(): void {
    this.map.clear();
  }
}
