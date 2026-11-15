// pixelsprite in the browser: the same sampler as pixelsprite/sampling.py, on onnxruntime-web.
//
// The ONNX graph (onnx/decoder.onnx in the Hugging Face repo) holds the 12 transformer blocks, the final norm and the
// head: one step with a key/value cache. Input vectors are sums of rows of small tables (onnx/embeddings.bin), built
// here exactly as in pixelsprite/model.py. Nothing in this file touches the page, so it also runs in Node (tests).

export const HF_REPO = "ihor-umanets/pixelsprite-canvas-v2";

export function modelFiles(base) {
  return {
    decoder: `${base}/onnx/decoder.onnx`,   // float32 graph, large weights as external data "decoder.data"
    decoderStatic: `${base}/onnx/decoder_static.onnx`,  // the same with a fixed-size cache (same external data)
    weights: `${base}/onnx/decoder.f16`,    // those weights as float16, in the same order
    embJson: `${base}/onnx/embeddings.json`,
    embBin: `${base}/onnx/embeddings.bin`,
    palette: `${base}/palette.json`,
  };
}

export const DATA_NAME = "decoder.data";

const json = (bytes) => JSON.parse(new TextDecoder().decode(bytes));

// On the CPU, onnxruntime-web computes inside session.run on this thread and resolves at once, so a loop of awaits
// never gives the browser a chance to paint (the page looks frozen; phones may give up on it). A MessageChannel
// task (no 4 ms clamp like setTimeout) lets it render between steps.
const yieldToBrowser = () => new Promise((resolve) => {
  const ch = new MessageChannel();
  ch.port1.onmessage = () => { ch.port1.close(); resolve(); };
  ch.port2.postMessage(0);
});

let HALF = null;
/** Little-endian float16 bytes -> float32 bytes (the weights are shipped as float16 to halve the download). */
export function halfToFloat(bytes) {
  if (!HALF) {
    HALF = new Float32Array(65536);
    for (let h = 0; h < 65536; h++) {
      const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, f = h & 1023;
      HALF[h] = e === 0 ? s * f * 2 ** -24 : e === 31 ? (f ? NaN : s * Infinity) : s * (1 + f / 1024) * 2 ** (e - 15);
    }
  }
  if (bytes.byteOffset % 2) bytes = bytes.slice();
  const h = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength >> 1), out = new Float32Array(h.length);
  for (let i = 0; i < h.length; i++) out[i] = HALF[h[i]];
  return new Uint8Array(out.buffer);
}

/** Loads the decoder. The weights go in as external data: onnxruntime-web then uploads them straight to the GPU
 * (WebGPU) instead of first copying the whole model into the WebAssembly heap, which never shrinks again; that
 * copy alone took ~560 MB and crashed phones. files.weights may be omitted for a self-contained graph (tests).
 * fixedCache: use decoder_static.onnx, whose cache keeps one size for the whole run (memory stays flat). */
export async function loadModel({ ort, files, ep = "wasm", fetchBytes, onProgress, sessionOptions = {}, fixedCache = false }) {
  const [info, pal, emb] = await Promise.all([
    fetchBytes(files.embJson).then(json), fetchBytes(files.palette).then(json), fetchBytes(files.embBin)]);
  const decoder = await fetchBytes(fixedCache ? files.decoderStatic : files.decoder, files.weights ? null : onProgress);
  const opts = { executionProviders: [ep], graphOptimizationLevel: "all", ...sessionOptions };
  if (files.weights) opts.externalData = [{ path: DATA_NAME, data: halfToFloat(await fetchBytes(files.weights, onProgress)) }];
  if (ep === "webgpu") {  // keep the cache on the GPU between steps; only the logits come back
    opts.preferredOutputLocation = {};
    for (let i = 0; i < info.n_layers; i++) {
      opts.preferredOutputLocation[`present_k_${i}`] = "gpu-buffer";
      opts.preferredOutputLocation[`present_v_${i}`] = "gpu-buffer";
    }
  }
  const session = await ort.InferenceSession.create(decoder, opts);
  const model = new SpriteModel(ort, session, info, emb, pal.colors, ep);
  model.fixedCache = fixedCache;
  return model;
}

// ------------------------------------------------------------------------------------------------- random numbers
export function makeRng(seed) {  // mulberry32: small, fast, seedable (the sprites differ from Python's for a seed)
  let a = (Number(seed) >>> 0) ^ 0x9e3779b9;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ------------------------------------------------------------------------------------------------------ guidance
function logSumExp(a, off, len) {
  let m = -Infinity;
  for (let i = 0; i < len; i++) if (a[off + i] > m) m = a[off + i];
  let s = 0;
  for (let i = 0; i < len; i++) s += Math.exp(a[off + i] - m);
  return m + Math.log(s);
}

function emptyLogOdds(l, off, V, E) {  // log p(EMPTY) - log(1 - p(EMPTY)), as split_pick
  const lp = l[off + E] - logSumExp(l, off, V);
  return lp - Math.log1p(-Math.min(Math.exp(lp), 1 - 1e-6));
}

/** Guided distribution for one pixel. lc / lu: logits with and without text at offsets offC / offU (offU < 0 = no
 * guidance). Fills `col` (length K) with colour probabilities after temperature and top-p; returns p(EMPTY). */
export function guided(lc, offC, lu, offU, K, gShape, gColour, temperature, topP, col, order) {
  const V = K + 1, E = K;
  let s;
  if (offU >= 0) {
    const sc = emptyLogOdds(lc, offC, V, E), su = emptyLogOdds(lu, offU, V, E);
    s = su + gShape * (sc - su);
    const lsC = logSumExp(lc, offC, K), lsU = logSumExp(lu, offU, K);
    for (let j = 0; j < K; j++) {
      const u = lu[offU + j] - lsU;
      col[j] = u + gColour * ((lc[offC + j] - lsC) - u);
    }
  } else {
    s = emptyLogOdds(lc, offC, V, E);
    for (let j = 0; j < K; j++) col[j] = lc[offC + j];
  }
  const inv = 1 / Math.max(temperature, 1e-6);
  let m = -Infinity;
  for (let j = 0; j < K; j++) { col[j] *= inv; if (col[j] > m) m = col[j]; }
  let sum = 0;
  for (let j = 0; j < K; j++) { col[j] = Math.exp(col[j] - m); sum += col[j]; }
  for (let j = 0; j < K; j++) col[j] /= sum;
  if (topP < 1) {  // nucleus: keep the most likely colours whose mass before them is < topP (always the first)
    for (let j = 0; j < K; j++) order[j] = j;
    order.sort((a, b) => col[b] - col[a]);
    let cum = 0, kept = 0;
    for (let r = 0; r < K; r++) {
      const j = order[r];
      if (cum < topP) { cum += col[j]; kept += col[j]; } else col[j] = 0;
    }
    for (let j = 0; j < K; j++) col[j] /= kept;
  }
  return 1 / (1 + Math.exp(-s));
}

function draw(col, K, u) {
  let acc = 0;
  for (let j = 0; j < K; j++) { acc += col[j]; if (u < acc) return j; }
  for (let j = K - 1; j >= 0; j--) if (col[j] > 0) return j;  // rounding at the very end
  return 0;
}

// ------------------------------------------------------------------------------------------------ silhouettes
/** Edge flag per pixel: 0 inside, 1 on the boundary (4-neighbourhood, canvas edge = outside), 2 outside,
 * 3 unknown (no silhouette given). */
export function edgeFlags(mask, S) {
  const e = new Int8Array(S * S);
  if (!mask) return e.fill(3);
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    const p = y * S + x;
    if (!mask[p]) { e[p] = 2; continue; }
    const inner = y > 0 && mask[p - S] && y < S - 1 && mask[p + S] && x > 0 && mask[p - 1] && x < S - 1 && mask[p + 1];
    e[p] = inner ? 0 : 1;
  }
  return e;
}

/** Sizes of the 8-connected pieces of the non-transparent pixels, largest first. */
export function pieces(img, S) {
  const seen = new Uint8Array(S * S), sizes = [], q = new Int32Array(S * S);
  for (let p0 = 0; p0 < S * S; p0++) {
    if (img[p0] < 0 || seen[p0]) continue;
    let head = 0, tail = 0, n = 0;
    q[tail++] = p0; seen[p0] = 1;
    while (head < tail) {
      const p = q[head++]; n++;
      const y = (p / S) | 0, x = p % S;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const yy = y + dy, xx = x + dx;
        if (yy < 0 || xx < 0 || yy >= S || xx >= S) continue;
        const pp = yy * S + xx;
        if (img[pp] >= 0 && !seen[pp]) { seen[pp] = 1; q[tail++] = pp; }
      }
    }
    sizes.push(n);
  }
  return sizes.sort((a, b) => b - a);
}

/** Reason to draw a sprite again (empty canvas or a detached piece of 1-2 px), or null. */
export function rejected(img, S, maxPx = 2) {
  const sizes = pieces(img, S);
  if (!sizes.length) return "empty";
  if (sizes.length >= 2 && Math.min(...sizes.slice(1)) <= maxPx) return "detached_1_2px";
  return null;
}

// ---------------------------------------------------------------------------------------------------- the model
export class SpriteModel {
  constructor(ort, session, info, embBytes, palette, ep) {
    this.ort = ort; this.session = session; this.info = info; this.ep = ep;
    const buf = embBytes.slice().buffer;  // own, aligned copy
    this.t = {};
    for (const [name, v] of Object.entries(info.tables))
      this.t[name] = new Float32Array(buf, v.offset, v.shape.reduce((a, b) => a * b, 1));
    this.d = info.d_model; this.K = info.n_colors; this.E = info.empty; this.BOS = info.bos;
    this.L = info.n_layers; this.H = info.n_heads; this.hd = this.d / this.H;
    this.textDim = info.tables["cond_proj.weight"].shape[1];
    this.palette = Uint8Array.from(palette.flat());
  }

  _add(dst, o, name, i) { const d = this.d, t = this.t[name]; for (let j = 0; j < d; j++) dst[o + j] += t[i * d + j]; }

  /** Caption vector (Float32Array, L2-normalised) -> condition vector; null -> the learned "no text" vector. */
  condition(text) {
    if (!text) return this.t.null_cond.slice();
    const W = this.t["cond_proj.weight"], b = this.t["cond_proj.bias"], D = this.textDim, out = new Float32Array(this.d);
    for (let o = 0; o < this.d; o++) {
      let s = b[o];
      for (let i = 0; i < D; i++) s += W[o * D + i] * text[i];
      out[o] = s;
    }
    return out;
  }

  /** Writes the 2 + S prefix vectors (text, size, one per silhouette row) at dst[o...]. */
  writePrefix(dst, o, cond, S, mask) {
    const d = this.d, k = this.info.kinds, M = this.info.max_size;
    dst.set(cond, o); this._add(dst, o, "kind", k.text);
    o += d;
    dst.fill(0, o, o + d); this._add(dst, o, "size_emb", this.info.sizes.indexOf(S)); this._add(dst, o, "kind", k.size);
    const W = this.t["mask_proj.weight"], bias = this.t["mask_proj.bias"];
    for (let r = 0; r < S; r++) {
      o += d;
      dst.set(bias, o);
      this._add(dst, o, "row", r);
      this._add(dst, o, "kind", mask ? k.row_known : k.row_unknown);
      if (mask) for (let c = 0; c < S; c++) if (mask[r * S + c]) for (let j = 0; j < d; j++) dst[o + j] += W[j * M + c];
    }
  }

  /** Writes one pixel token: previous pixel's token + row + column + edge flag of this pixel. */
  writePixel(dst, o, prev, r, c, e) {
    dst.fill(0, o, o + this.d);
    this._add(dst, o, "tok", prev); this._add(dst, o, "row", r); this._add(dst, o, "col", c);
    this._add(dst, o, "edge", e); this._add(dst, o, "kind", this.info.kinds.pixel);
  }

  emptyCache(B) {
    const z = new Float32Array(0), out = [];
    for (let i = 0; i < 2 * this.L; i++) out.push(new this.ort.Tensor("float32", z, [B, this.H, 0, this.hd]));
    return out;
  }

  /** One decoder call: x (B*T*d) + cache -> logits of the last token (B*(K+1)) and the longer cache. */
  async step(x, B, T, cache) {
    const feeds = { x: new this.ort.Tensor("float32", x, [B, T, this.d]) };
    for (let i = 0; i < this.L; i++) { feeds[`past_k_${i}`] = cache[2 * i]; feeds[`past_v_${i}`] = cache[2 * i + 1]; }
    const out = await this.session.run(feeds);
    const next = [];
    for (let i = 0; i < this.L; i++) next.push(out[`present_k_${i}`], out[`present_v_${i}`]);
    return { logits: out.logits.data, cache: next };
  }

  static release(cache) { for (const t of cache) if (t.location === "gpu-buffer") t.dispose(); }

  /** A zero-filled fixed-size cache: 2 * layers tensors (B, H, M, hd). On WebGPU they are created on the GPU
   * (new buffers start as zeros): a CPU tensor would first be copied into the WebAssembly heap, which never shrinks
   * (256 MB after one 16x16 run of 8 sprites, 531 MB after 32x32). */
  async cacheSet(B, M) {
    const dims = [B, this.H, M, this.hd], out = [];
    if (this.ep === "webgpu") {
      const device = await this.ort.env.webgpu.device, size = B * this.H * M * this.hd * 4;
      for (let i = 0; i < 2 * this.L; i++) {
        const buf = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        out.push(this.ort.Tensor.fromGpuBuffer(buf, { dataType: "float32", dims, dispose: () => buf.destroy() }));
      }
      return out;
    }
    for (let i = 0; i < 2 * this.L; i++) out.push(new this.ort.Tensor("float32", new Float32Array(B * this.H * M * this.hd), dims));
    return out;
  }

  /** One call of decoder_static.onnx: token x (B*d) at position pos. `into` (CPU only): a cache set the graph writes
   * its new cache into, so no new arrays are made per step. */
  async stepFixed(x, B, pos, cache, into) {
    const feeds = { x: new this.ort.Tensor("float32", x, [B, 1, this.d]),
                    pos: new this.ort.Tensor("int64", BigInt64Array.of(BigInt(pos)), [1]) };
    for (let i = 0; i < this.L; i++) { feeds[`past_k_${i}`] = cache[2 * i]; feeds[`past_v_${i}`] = cache[2 * i + 1]; }
    let out;
    if (into) {
      const fetches = { logits: null };
      for (let i = 0; i < this.L; i++) { fetches[`present_k_${i}`] = into[2 * i]; fetches[`present_v_${i}`] = into[2 * i + 1]; }
      out = await this.session.run(feeds, fetches);
    } else out = await this.session.run(feeds);
    const next = [];
    for (let i = 0; i < this.L; i++) next.push(out[`present_k_${i}`], out[`present_v_${i}`]);
    return { logits: out.logits.data, cache: next };
  }

  /** Draws b sprites on an S x S canvas in parallel. Returns b Int16Arrays (palette index, -1 = transparent). */
  async drawChunk(p, b, rng, onPixel) {
    const { condC, condU, S, mask, win, gShape, gColour, temperature, topP } = p;
    const rep = condU ? 2 : 1, B = b * rep, n = S * S, d = this.d, K = this.K, E = this.E, V = K + 1;
    const edge = edgeFlags(mask, S);
    const T0 = 2 + S + 1, x0 = new Float32Array(B * T0 * d);
    for (let j = 0; j < B; j++) {
      this.writePrefix(x0, j * T0 * d, j >= b ? condU : condC, S, mask);
      this.writePixel(x0, (j * T0 + T0 - 1) * d, this.BOS, 0, 0, edge[0]);
    }
    let logits, cache, spare = null, pos = 0, lastYield = performance.now();
    const advance = async (xt) => {  // one more token -> logits of the next pixel
      const prev = cache;
      let r;
      if (!this.fixedCache) r = await this.step(xt, B, 1, cache);
      else if (spare) {  // CPU: write into the spare set, then swap; if onnxruntime ignored it, stop using spares
        const into = spare;
        r = await this.stepFixed(xt, B, pos, cache, into);
        spare = r.cache[0] === into[0] ? prev : null;
      }
      else r = await this.stepFixed(xt, B, pos, cache, null);
      pos++;
      logits = r.logits; cache = r.cache;
      if (prev !== cache) SpriteModel.release(prev);
    };
    if (this.fixedCache) {  // fixed-size cache: the prefix goes in one token at a time, like the pixels
      cache = await this.cacheSet(B, 2 + S + n);
      if (this.ep !== "webgpu") spare = await this.cacheSet(B, 2 + S + n);
      const xt = new Float32Array(B * d);
      for (let i = 0; i < T0; i++) {
        for (let j = 0; j < B; j++) xt.set(x0.subarray((j * T0 + i) * d, (j * T0 + i + 1) * d), j * d);
        await advance(xt);
      }
    } else ({ logits, cache } = await this.step(x0, B, T0, this.emptyCache(B)));
    const out = Array.from({ length: b }, () => new Int16Array(n).fill(-1));
    const x = new Float32Array(B * d), tok = new Int32Array(b);
    const col = new Float64Array(K), order = topP < 1 ? new Int32Array(K) : null;
    try {
      for (let t = 0; t < n; t++) {
        if (p.signal && p.signal.aborted) { const e = new Error("stopped"); e.name = "AbortError"; throw e; }
        const forceEmpty = !win[t] || (mask && !mask[t]), forceFill = mask && mask[t];
        for (let i = 0; i < b; i++) {
          let pe = 0;
          if (forceEmpty) pe = 1;
          else {
            pe = guided(logits, i * V, logits, condU ? (b + i) * V : -1, K, gShape, gColour, temperature, topP, col, order);
            if (forceFill) pe = 0;
          }
          tok[i] = rng() < pe ? E : draw(col, K, rng());
          out[i][t] = tok[i] === E ? -1 : tok[i];
        }
        if (onPixel) onPixel(t, out);
        if (t + 1 === n) break;
        const r = ((t + 1) / S) | 0, c = (t + 1) % S;
        for (let j = 0; j < B; j++) this.writePixel(x, j * d, tok[j % b], r, c, edge[t + 1]);
        await advance(x);
        if (performance.now() - lastYield > 40) { await yieldToBrowser(); lastYield = performance.now(); }
      }
    } finally {
      SpriteModel.release(cache);
    }
    return out;
  }

  /** n sprites of size 16, 24 or 32 (24 = a 24x24 window in the middle of the 32x32 canvas).
   * vector: caption vector or null; mask: Uint8Array(size*size) silhouette or null (then the filter is off).
   * onPixel(ids, t, images, view) is called after every pixel; view = {S, o, w}: canvas size, window offset and
   * width. signal: an AbortSignal to stop between pixels. Returns {images: Int16Array(size*size)[], stats}. */
  async generate({ vector = null, n = 8, size = 16, seed = 0, gShape = 1, gColour = 2, temperature = 1, topP = 1,
                   mask = null, filter = true, chunk = 8, maxRounds = 50, onPixel = null, onSprite = null,
                   signal = null }) {
    if (![16, 24, 32].includes(size)) throw new Error("size must be 16, 24 or 32");
    const S = size === 24 ? 32 : size, w = size, o = (S - w) >> 1;
    let m = null;
    if (mask) {
      m = new Uint8Array(S * S);
      for (let y = 0; y < w; y++) for (let x = 0; x < w; x++) m[(y + o) * S + x + o] = mask[y * w + x] ? 1 : 0;
    }
    const win = new Uint8Array(S * S);
    for (let y = o; y < o + w; y++) for (let x = o; x < o + w; x++) win[y * S + x] = 1;
    const useFilter = filter && !mask;
    const condC = this.condition(vector);
    const condU = vector && (gShape !== 1 || gColour !== 1) ? this.condition(null) : null;
    const p = { condC, condU, S, mask: m, win, gShape, gColour, temperature, topP, signal };
    const rng = makeRng(seed), view = { S, o, w };
    const images = new Array(n), stats = { sampled: 0, rejected: 0, rounds: 0, reasons: {} };
    let pending = Array.from({ length: n }, (_, i) => i);
    while (pending.length && stats.rounds < maxRounds) {
      stats.rounds++;
      const again = [];
      for (let s = 0; s < pending.length; s += chunk) {
        const ids = pending.slice(s, s + chunk);
        const got = await this.drawChunk(p, ids.length, rng, onPixel ? (t, imgs) => onPixel(ids, t, imgs, view) : null);
        ids.forEach((id, k) => {
          stats.sampled++;
          const why = useFilter ? rejected(got[k], S) : null;
          if (why) { stats.rejected++; stats.reasons[why] = (stats.reasons[why] || 0) + 1; again.push(id); return; }
          images[id] = crop(got[k], S, o, w);
          if (onSprite) onSprite(id, images[id]);
        });
      }
      pending = again;
    }
    if (pending.length) throw new Error(`${pending.length} sprites still rejected after ${maxRounds} rounds`);
    return { images, size, stats };
  }

  /** Palette-index image -> RGBA bytes (alpha 0 or 255). */
  rgba(img) {
    const out = new Uint8ClampedArray(img.length * 4);
    for (let p = 0; p < img.length; p++) {
      const k = img[p];
      if (k < 0) continue;
      out[4 * p] = this.palette[3 * k]; out[4 * p + 1] = this.palette[3 * k + 1]; out[4 * p + 2] = this.palette[3 * k + 2];
      out[4 * p + 3] = 255;
    }
    return out;
  }
}

function crop(img, S, o, w) {
  if (w === S) return img;
  const out = new Int16Array(w * w);
  for (let y = 0; y < w; y++) for (let x = 0; x < w; x++) out[y * w + x] = img[(y + o) * S + x + o];
  return out;
}
