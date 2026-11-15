// The demo page: loads onnxruntime-web and the model, turns captions into vectors, paints sprites as they are drawn.
// Preview build at /next/: own files, own model cache and own localStorage keys, so it never touches the main demo.
import { HF_REPO, modelFiles, loadModel } from "./engine.js?v=6";  // ?v=N: bump with every release so browsers never mix cached files of two versions
import { ENCODER, encoderFiles, loadTextEncoder } from "./text.js?v=6";

const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort.webgpu.min.mjs";
const TOKENIZERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/tokenizers@0.2.0/dist/tokenizers.min.mjs";
const MODEL_REVISION = "0a29494e9fc020a26feb6f97ec2047fc207ce898";  // a commit, so cached files never go stale
const CACHE_NAME = "pixelsprite-next";

const params = new URLSearchParams(location.search);
const base = (v, dflt) => (v && /^(\/|\.\/|https:\/\/)/.test(v) ? v.replace(/\/$/, "") : dflt);  // dev: ?model=/path
const MODEL_BASE = base(params.get("model"), `https://huggingface.co/${HF_REPO}/resolve/${MODEL_REVISION}`);
const ENCODER_BASE = base(params.get("encoder"), `https://huggingface.co/${ENCODER.repo}/resolve/${ENCODER.revision}`);

const $ = (id) => document.getElementById(id);
const radio = (name) => document.querySelector(`input[name=${name}]:checked`).value;
const state = { ort: null, model: null, ep: null, encoder: null, presets: new Map(), running: null, pad: null, padSize: 16 };

// Phones (iOS Safari / Chrome, Android) give a tab far less memory than a desktop: draw fewer sprites at once there.
const MOBILE = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent) ||
  (navigator.maxTouchPoints > 1 && /Macintosh/.test(navigator.userAgent));  // iPadOS reports itself as a Mac
// iPhone / iPad (every iOS browser is WebKit): WebGPU there crashed the tab or lost the device mid-run in tests
// (onnxruntime-web 1.31, Safari 26.6), so iOS draws on the CPU unless ?ep=webgpu. ?ios simulates it on a desktop.
const IOS = params.has("ios") || /iPhone|iPad|iPod/.test(navigator.userAgent) ||
  (navigator.maxTouchPoints > 1 && /Macintosh/.test(navigator.userAgent));

// If the tab dies while drawing (out of memory), the browser reloads the page. A marker left in localStorage tells
// the next load to be more careful: level 1 draws one sprite at a time, level 2 also leaves the GPU for the CPU.
// The marker is removed when a run ends and on "pagehide" (reload, closing the tab, leaving the page), which a crash
// never fires, so only a real crash counts.
const store = {
  get: (k) => { try { return JSON.parse(localStorage.getItem(k)); } catch { return null; } },
  set: (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode */ } },
  del: (k) => { try { localStorage.removeItem(k); } catch { /* private mode */ } },
};
const BUSY = "pixelsprite-next-busy", SAFE = "pixelsprite-next-safe";
function safeLevel() {
  if (params.get("safe") === "0") { store.del(SAFE); store.del(BUSY); return 0; }
  let level = Number(store.get(SAFE)) || 0;
  const busy = store.get(BUSY);
  if (busy && Date.now() - busy.t < 30 * 60e3) {
    level = Math.min(2, Math.max(level, busy.level || 0) + 1);
    store.set(SAFE, level);
    state.crashed = true;
    state.crashedAt = `${busy.ep === "webgpu" ? "WebGPU" : "CPU"}${busy.at ? `, ${busy.at}` : ""}`;
  }
  store.del(BUSY);
  return level;
}
// The marker also records how far the run got (updated about once a second), so after a crash the diagnostics
// line can say where it died: at the same pixel every time, or later with fewer sprites.
const markBusy = (phase, at = "") => store.set(BUSY, { t: Date.now(), phase, level: state.safe, ep: state.ep, at });
const markIdle = () => store.del(BUSY);
addEventListener("pagehide", markIdle);

// ------------------------------------------------------------------------------------------- downloads + cache
async function fetchBytes(url, onProgress) {
  let cache = null;
  try {
    cache = await caches.open(CACHE_NAME);
    const hit = await cache.match(url);
    if (hit) return new Uint8Array(await hit.arrayBuffer());
  } catch { cache = null; }
  const r = await fetch(url);
  if (!r.ok) throw new Error(`download failed (${r.status}): ${url}`);
  const total = Number(r.headers.get("content-length")) || 0;
  let bytes;
  if (onProgress && r.body && total) {  // straight into one buffer: no second copy of 77 MB
    const reader = r.body.getReader();
    bytes = new Uint8Array(total);
    let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (got + value.length > total) throw new Error(`download larger than announced: ${url}`);
      bytes.set(value, got); got += value.length; onProgress(got, total);
    }
    if (got !== total) throw new Error(`download cut short: ${url}`);
  } else bytes = new Uint8Array(await r.arrayBuffer());
  if (cache) try { await cache.put(url, new Response(bytes)); } catch { /* storage full or blocked: fine */ }
  return bytes;
}

async function pruneCache() {  // drop files of older model / encoder revisions
  try {
    const cache = await caches.open(CACHE_NAME);
    for (const req of await cache.keys())
      if (!req.url.startsWith(MODEL_BASE) && !req.url.startsWith(ENCODER_BASE)) await cache.delete(req);
  } catch { /* no Cache Storage */ }
}

// ------------------------------------------------------------------------------------------------- status line
function status(text, { error = false, frac = null } = {}) {
  $("statusText").textContent = text;
  $("status").classList.toggle("error", error);
  $("bar").hidden = frac === null;
  if (frac !== null) $("barFill").style.width = `${Math.round(frac * 100)}%`;
}
const mb = (b) => (b / 1e6).toFixed(0);
const progress = (what) => (got, total) =>
  status(total ? `${what}: ${mb(got)} of ${mb(total)} MB` : `${what}: ${mb(got)} MB`, { frac: total ? got / total : 0 });

const readyText = () => {
  let s = state.ep === "webgpu" ? "Ready. Drawing on your GPU."
    : state.safe >= 2 ? "Ready. Drawing on the CPU."
    : state.why === "no WebGPU in this browser"
      ? "Ready. No WebGPU in this browser, so drawing on the CPU: 16×16 is fine, 24×24 and 32×32 are slow."
      : "Ready. Drawing on the CPU: 16×16 is fine, 24×24 and 32×32 are slow.";
  if (state.crashed) s += " The page closed during the last run, probably out of memory, so it now draws one sprite at a time"
    + (state.safe >= 2 ? " on the CPU." : ".");
  return s;
};

// ------------------------------------------------------------------------------------------------ model loading
async function pickEP() {
  const forced = params.get("ep");
  if (forced === "wasm" || forced === "webgpu") { state.why = `forced by ?ep=${forced}`; return forced; }
  if (state.safe >= 2) { state.why = "CPU after crashes or a WebGPU failure"; return "wasm"; }
  if (IOS) { state.why = "iPhone / iPad: CPU by default (?ep=webgpu to try the GPU)"; return "wasm"; }
  if (!navigator.gpu) { state.why = "no WebGPU in this browser"; return "wasm"; }
  try {
    if (await navigator.gpu.requestAdapter()) { state.why = "WebGPU adapter found"; return "webgpu"; }
    state.why = "WebGPU present but no adapter";
  } catch (e) { state.why = `WebGPU adapter error: ${e.message || e}`; }
  return "wasm";
}

async function warmUp(model) {  // two steps with a tiny fixed cache: compiles shaders and proves the backend works
  const x = new Float32Array(model.d);
  model.writePixel(x, 0, model.BOS, 0, 0, 3);
  const zero = await model.cacheSet(1, 4);
  const a = await model.stepFixed(x, 1, 0, zero, null);
  model.constructor.release(zero);
  model.writePixel(x, 0, model.E, 0, 1, 3);
  const b = await model.stepFixed(x, 1, 1, a.cache, null);
  model.constructor.release(a.cache); model.constructor.release(b.cache);
  if (!Number.isFinite(b.logits[0])) throw new Error("the backend returned NaN");
}

function diagnostics() {  // one line for screenshots: what this device got and why
  const ua = navigator.userAgent, ios = ua.match(/OS (\d+)[._](\d+)/), saf = ua.match(/Version\/(\d+(\.\d+)?)/);
  const parts = [state.ep === "webgpu" ? "WebGPU" : "CPU", state.why, MOBILE ? "phone" : "desktop",
                 `batch ${chunkFor(16)} / ${chunkFor(32)}`, `safe ${state.safe}`];
  if (/iPhone|iPad|iPod/.test(ua) && ios) parts.push(`iOS ${ios[1]}.${ios[2]}`);
  if (saf) parts.push(`Safari ${saf[1]}`);
  parts.push(`heap ${heapMB()} MB`);
  if (state.crashedAt) parts.push(`last crash: ${state.crashedAt}`);
  $("diag").textContent = parts.join(" · ");
}

// Record WebAssembly memories, to show how big onnxruntime's heap is in the diagnostics line and in the crash
// marker (phones give no other memory reading). Installed before onnxruntime is imported.
const wasmMems = new Set();
{
  const Orig = WebAssembly.Memory;
  WebAssembly.Memory = new Proxy(Orig, { construct(t, a) { const m = new t(...a); wasmMems.add(m); return m; } });
  const grab = (r) => { for (const v of Object.values((r.instance || r).exports || {})) if (v instanceof Orig) wasmMems.add(v); return r; };
  const inst = WebAssembly.instantiate, stream = WebAssembly.instantiateStreaming;
  WebAssembly.instantiate = (...a) => inst.apply(WebAssembly, a).then(grab);
  if (stream) WebAssembly.instantiateStreaming = (...a) => stream.apply(WebAssembly, a).then(grab);
}
const heapMB = () => Math.max(0, ...[...wasmMems].map((m) => Math.round(m.buffer.byteLength / 1e6)));
window.wasmMB = () => [...wasmMems].map((m) => Math.round(m.buffer.byteLength / 1e6));

async function loadBackend(ep) {
  const m = await loadModel({ ort: state.ort, files: modelFiles(MODEL_BASE), ep, fetchBytes, fixedCache: true,
                              onProgress: progress("Downloading the model (once)") });
  status(ep === "webgpu" ? "Preparing the GPU…" : "Preparing…");
  await warmUp(m);
  return m;
}

function setBackend(ep) {
  state.ep = ep;
  $("ep").textContent = ep === "webgpu" ? "WebGPU" : "CPU";
  $("ep").hidden = false;
  if (!nTouched) $("n").value = defaultN(Number(radio("size")));
  diagnostics();
  status(readyText());
  updateHints();
}

/** WebGPU broke in the middle of a run (e.g. "map async was not successful": the GPU device was lost). Load the
 * CPU backend instead (the files are cached) and remember it for this device. */
async function switchToCPU(reason) {
  state.why = `WebGPU failed while drawing: ${reason}`.slice(0, 160);
  state.safe = 2;
  store.set(SAFE, 2);
  const old = state.model;
  state.model = null;
  $("go").disabled = true;
  try { await old?.session.release(); } catch { /* already broken */ }
  status("Switching to the CPU…");
  state.model = await loadBackend("wasm");
  setBackend("wasm");
  $("go").disabled = false;
}

async function boot() {
  try {
    state.safe = safeLevel();
    const [ort, presets] = await Promise.all([import(ORT_URL), fetch("demo/presets.json").then((r) => r.json())]);
    state.ort = ort;
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
    for (const it of presets.items) state.presets.set(it.text, b64floats(it.vector));
    buildChips(presets.items.map((i) => i.text));
    let ep = await pickEP();
    try {  // loading is not guarded: the CPU fallback needs more memory than WebGPU, not less
      state.model = await loadBackend(ep);
    } catch (e) {
      if (ep !== "webgpu") throw e;
      console.warn("WebGPU failed, falling back to the CPU:", e);
      state.why = `WebGPU failed: ${e.message || e}`.slice(0, 160);
      ep = "wasm";
      state.model = await loadBackend(ep);
    }
    setBackend(ep);
    $("go").disabled = false;
    pruneCache();
    if (params.has("debug")) window.pixelsprite = state;  // for timing from the console
  } catch (e) {
    console.error(e);
    status(`Could not load the model: ${e.message || e}`, { error: true });
  }
}

function b64floats(s) {
  const bin = atob(s), u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(u8.buffer);
}

async function vectorFor(text) {
  if (!text) return null;
  if (state.presets.has(text)) return state.presets.get(text);
  if (!state.encoder) {
    state.encoder = (async () => {
      const { Tokenizer } = await import(TOKENIZERS_URL);
      return loadTextEncoder({ ort: state.ort, Tokenizer, files: encoderFiles(ENCODER_BASE), fetchBytes,
                               onProgress: progress("Downloading the text encoder (once)") });
    })();
    state.encoder.catch(() => { state.encoder = null; });
  }
  const enc = await state.encoder;
  status(readyText());
  return enc.encode(text);
}

// ---------------------------------------------------------------------------------------------------- controls
function buildChips(texts) {
  for (const t of [...texts, ""]) {
    const b = document.createElement("button");
    b.type = "button"; b.className = "chip"; b.textContent = t || "no text";
    b.onclick = () => { $("text").value = t; $("text").focus(); };
    $("chips").appendChild(b);
  }
}
for (const [id, out] of [["gs", "gsv"], ["gc", "gcv"], ["tc", "tcv"], ["tp", "tpv"]]) {
  const upd = () => { $(out).textContent = Number($(id).value).toFixed(2); };
  $(id).addEventListener("input", upd); upd();
}
$("rnd").onclick = () => { $("seed").value = Math.floor(Math.random() * 1e6); $("autoseed").checked = false; };
$("seed").addEventListener("input", () => { $("autoseed").checked = false; });

const MODE_HINTS = {
  new: "The model draws every sprite from scratch.",
  same: "One sprite first, then new colourings of its exact silhouette.",
  draw: "Draw a silhouette and the model colours it from the caption.",
};
function updateHints() {
  const size = Number(radio("size"));
  const hints = [];
  if (size === 24) hints.push("24×24 is a window in the middle of the 32×32 canvas: the model never trained on this size.");
  if (state.ep === "wasm" && size !== 16) hints.push("On the CPU this size takes about 40 s per sprite.");
  $("sizeHint").textContent = hints.join(" ");
  $("sizeHint").classList.toggle("warn", state.ep === "wasm" && size !== 16);
  const md = radio("mode");
  $("modeHint").textContent = MODE_HINTS[md];
  $("drawbox").hidden = md !== "draw";
  if (md === "draw" && state.padSize !== size) newPad(size);
}
document.querySelectorAll("input[name=mode], input[name=size]").forEach((r) => r.addEventListener("change", updateHints));
// Until the user sets "How many", it follows the size: 8 at 16x16 and 4 at 24x24 / 32x32 (4 and 2 on phones).
let nTouched = false;
const defaultN = (size) => (size === 16 ? (MOBILE ? (state.ep === "wasm" ? 2 : 4) : 8) : (MOBILE ? (state.ep === "wasm" ? 1 : 2) : 4));
$("n").value = defaultN(Number(radio("size")));
$("n").addEventListener("input", () => { nTouched = true; });
document.querySelectorAll("input[name=size]").forEach((r) => r.addEventListener("change", () => {
  if (!nTouched) $("n").value = defaultN(Number(radio("size")));
}));

// ------------------------------------------------------------------------------------------------- drawing pad
function newPad(size) { state.padSize = size; state.pad = new Uint8Array(size * size); drawPad(); }
function drawPad() {
  const cv = $("pad"), ctx = cv.getContext("2d"), s = state.padSize, cell = cv.width / s;
  const css = getComputedStyle(document.documentElement);
  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.fillStyle = css.getPropertyValue("--paint").trim();
  for (let y = 0; y < s; y++) for (let x = 0; x < s; x++) if (state.pad[y * s + x]) ctx.fillRect(x * cell, y * cell, cell, cell);
  ctx.strokeStyle = css.getPropertyValue("--grid").trim(); ctx.lineWidth = 1;
  for (let i = 0; i <= s; i++) {
    ctx.beginPath(); ctx.moveTo(i * cell + 0.5, 0); ctx.lineTo(i * cell + 0.5, cv.height); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, i * cell + 0.5); ctx.lineTo(cv.width, i * cell + 0.5); ctx.stroke();
  }
}
let painting = null, lastCell = null;
function padCell(e) {
  const r = $("pad").getBoundingClientRect(), s = state.padSize;
  const x = Math.floor(((e.clientX - r.left) / r.width) * s), y = Math.floor(((e.clientY - r.top) / r.height) * s);
  return x >= 0 && y >= 0 && x < s && y < s ? [x, y] : null;
}
function paintLine(a, b) {  // every cell between two pointer events, so fast strokes leave no gaps
  const n = Math.max(Math.abs(b[0] - a[0]), Math.abs(b[1] - a[1]), 1);
  for (let i = 0; i <= n; i++) {
    const x = Math.round(a[0] + ((b[0] - a[0]) * i) / n), y = Math.round(a[1] + ((b[1] - a[1]) * i) / n);
    state.pad[y * state.padSize + x] = painting ? 1 : 0;
  }
}
$("pad").addEventListener("contextmenu", (e) => e.preventDefault());
$("pad").addEventListener("pointerdown", (e) => {
  painting = !(e.button === 2 || e.shiftKey); $("pad").setPointerCapture(e.pointerId);
  const c = padCell(e); if (c) { paintLine(c, c); lastCell = c; drawPad(); }
});
$("pad").addEventListener("pointermove", (e) => {
  if (painting === null) return;
  const c = padCell(e); if (!c) return;
  paintLine(lastCell || c, c); lastCell = c; drawPad();
});
$("pad").addEventListener("pointerup", () => { painting = null; lastCell = null; });
$("clearPad").onclick = () => newPad(state.padSize);
$("fillPad").onclick = () => { state.pad.fill(1); drawPad(); };
$("invertPad").onclick = () => { state.pad = state.pad.map((v) => 1 - v); drawPad(); };
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => state.pad && drawPad());
newPad(16); updateHints();

// ----------------------------------------------------------------------------------------------------- results
function slug(t) { return (t || "no_text").toLowerCase().replace(/[^\p{L}\p{N}]+/gu, "_").replace(/^_|_$/g, "").slice(0, 40) || "no_text"; }

function makeCard(grid, w, label) {
  const card = document.createElement("div"); card.className = "card";
  const cv = document.createElement("canvas"); cv.className = "sprite"; cv.width = w; cv.height = w;
  cv.setAttribute("role", "img"); cv.setAttribute("aria-label", label);
  card.appendChild(cv);
  const acts = document.createElement("div"); acts.className = "acts"; card.appendChild(acts);
  grid.appendChild(card);
  const ctx = cv.getContext("2d");
  return { card, cv, ctx, acts, data: ctx.createImageData(w, w), img: null, w };
}

function paint(c, img, view, cursor) {  // img: palette indices on the S x S canvas (view) or the final w x w image
  const { S, o, w } = view, d = c.data.data, pal = state.model.palette;
  for (let y = 0; y < w; y++) for (let x = 0; x < w; x++) {
    const k = img[(y + o) * S + x + o], p = (y * w + x) * 4;
    if (k < 0) { d[p + 3] = 0; continue; }
    d[p] = pal[3 * k]; d[p + 1] = pal[3 * k + 1]; d[p + 2] = pal[3 * k + 2]; d[p + 3] = 255;
  }
  c.ctx.putImageData(c.data, 0, 0);
  if (cursor >= 0) {
    const y = ((cursor / S) | 0) - o, x = (cursor % S) - o;
    if (y >= 0 && x >= 0 && y < w && x < w) {
      c.ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--cursor");
      c.ctx.fillRect(x, y, 1, 1);
    }
  }
}

function finishCard(c, run, k) {
  const a = document.createElement("a"); a.className = "btn-sm"; a.textContent = "PNG";
  a.href = c.cv.toDataURL("image/png");
  a.download = `${slug(run.text)}_${run.size}_s${run.seed}_${String(k).padStart(2, "0")}.png`;
  const b = document.createElement("button"); b.type = "button"; b.className = "btn-sm"; b.textContent = "shape →";
  b.title = "Use this silhouette: colour it again from any caption";
  b.onclick = () => {
    document.querySelector(`input[name=size][value="${run.size}"]`).checked = true;
    document.querySelector("input[name=mode][value=draw]").checked = true;
    updateHints();
    state.pad = Uint8Array.from(c.img, (v) => (v >= 0 ? 1 : 0)); state.padSize = run.size; drawPad();
    $("text").focus();
    $("form").scrollIntoView({ behavior: "smooth", block: "start" });
  };
  c.acts.replaceChildren(a, b);
}

function sheetUrl(cards, w) {
  const cols = Math.min(8, cards.length), sc = Math.max(2, Math.round(128 / w)), cell = w * sc, gap = 8;
  const rows = Math.ceil(cards.length / cols);
  const cv = document.createElement("canvas");
  cv.width = cols * cell + (cols + 1) * gap; cv.height = rows * cell + (rows + 1) * gap;
  const ctx = cv.getContext("2d"); ctx.imageSmoothingEnabled = false;
  cards.forEach((c, k) => ctx.drawImage(c.cv, gap + (k % cols) * (cell + gap), gap + Math.floor(k / cols) * (cell + gap), cell, cell));
  return cv.toDataURL("image/png");
}

// ------------------------------------------------------------------------------------------------------- a run
function readSettings() {
  if ($("autoseed").checked) $("seed").value = Math.floor(Math.random() * 1e6);
  const n = Math.max(1, Math.min(32, Math.round(Number($("n").value) || 1)));
  $("n").value = n;
  return {
    text: $("text").value.trim(), size: Number(radio("size")), mode: radio("mode"), n, seed: Math.round(Number($("seed").value) || 0),
    gShape: Number($("gs").value), gColour: Number($("gc").value), temperature: Number($("tc").value),
    topP: Number($("tp").value), filter: $("filter").checked,
  };
}

// Sprites drawn in parallel. A step costs about the same for 1 or 16 rows on WebGPU, but the key/value cache grows
// with batch x length: 8 guided 16x16 sprites need ~220 MB of GPU memory at the end, 4 guided 32x32 ones ~420 MB.
// On the CPU every step also copies the cache in and out of WebAssembly. Phones get 2 / 1, after a crash 1.
function chunkFor(size) {
  const big = size !== 16;
  if (state.safe >= 1) return 1;
  if (MOBILE) return 1;
  return state.ep === "webgpu" ? (big ? 4 : 8) : (big ? 2 : 8);
}

async function run(st) {
  $("empty")?.remove();
  const el = document.createElement("article"); el.className = "panel run";
  const head = document.createElement("div"); head.className = "run-head";
  const title = document.createElement("div"); title.className = "run-title";
  const tt = document.createElement("span"); tt.textContent = st.text ? `“${st.text}”` : "no text";
  const badge = document.createElement("span"); badge.className = "badge"; badge.textContent = `${st.size}×${st.size}`;
  title.append(tt, badge);
  const sheetBtn = document.createElement("button"); sheetBtn.type = "button"; sheetBtn.className = "btn-sm"; sheetBtn.textContent = "download sheet"; sheetBtn.hidden = true;
  head.append(title, sheetBtn);
  const meta = document.createElement("div"); meta.className = "meta"; meta.textContent = "preparing…";
  const grid = document.createElement("div"); grid.className = "grid";
  el.append(head, meta, grid);
  $("results").prepend(el);

  const ctrl = new AbortController();
  state.running = ctrl;
  markBusy("run");
  const go = $("go"); go.textContent = "Stop"; go.classList.add("stop");
  const t0 = performance.now();
  let frame = 0;  // pending progress repaint; cancelled on error so it cannot overwrite the error message
  try {
    const vector = await vectorFor(st.text);
    const model = state.model, w = st.size, chunk = chunkFor(st.size);
    const common = { vector, size: st.size, gShape: st.gShape, gColour: st.gColour, temperature: st.temperature, topP: st.topP,
                     chunk, signal: ctrl.signal };
    let mask = null, source = null, seed = st.seed;
    if (st.mode === "draw") {
      if (state.padSize !== st.size) newPad(st.size);
      if (!state.pad.some((v) => v)) throw new Error("Draw a silhouette first.");
      mask = state.pad.slice();
    }
    let pending = null, done = 0, total = 0, noted = 0;
    const live = (cards) => (ids, t, imgs, view) => {
      pending = { ids, t, imgs, view, cards };
      const now = performance.now();
      if (now - noted > 1000) {
        noted = now;
        markBusy("run", `${st.size}×${st.size}, sprite ${done + 1} of ${total}, pixel ${t + 1} of ${view.S * view.S}, ${((now - t0) / 1000).toFixed(0)} s, heap ${heapMB()} MB`);
      }
      if (!frame) frame = requestAnimationFrame(() => {
        frame = 0;
        const p = pending;
        p.ids.forEach((id, k) => paint(p.cards[id], p.imgs[k], p.view, p.t + 1 < p.view.S * p.view.S ? p.t + 1 : -1));
        const secs = ((performance.now() - t0) / 1000).toFixed(1);
        meta.textContent = `drawing pixel ${p.t + 1} of ${p.view.S * p.view.S} · ${done} of ${total} sprites done · ${secs} s`;
      });
    };
    if (st.mode === "same") {
      total = 1;
      source = makeCard(grid, w, "source sprite");
      source.card.classList.add("source");
      const r = await model.generate({ ...common, n: 1, seed, filter: st.filter, chunk: 1, onPixel: live([source]) });
      source.img = r.images[0];
      paint(source, source.img, { S: w, o: 0, w }, -1);
      const tag = document.createElement("span"); tag.className = "tag"; tag.textContent = "source shape";
      source.acts.replaceChildren(tag);
      mask = Uint8Array.from(source.img, (v) => (v >= 0 ? 1 : 0));
      seed += 1;
    }
    const cards = Array.from({ length: st.n }, (_, k) => makeCard(grid, w, `sprite ${k + 1}`));
    done = 0; total = st.n;
    const res = await model.generate({
      ...common, n: st.n, seed, mask, filter: st.filter, onPixel: live(cards),
      onSprite: (id, img) => { cards[id].img = img; paint(cards[id], img, { S: w, o: 0, w }, -1); done++; },
    });
    if (frame) cancelAnimationFrame(frame);
    const all = source ? [source, ...cards] : cards;
    cards.forEach((c, k) => { c.img = res.images[k]; paint(c, c.img, { S: w, o: 0, w }, -1); finishCard(c, st, k); });
    if (source) finishCard(source, st, "src");
    const secs = (performance.now() - t0) / 1000;
    const modeName = { new: "new shapes", same: "one shape", draw: "your shape" }[st.mode];
    let m = `${modeName} · seed ${st.seed} · shape g ${st.gShape} · colour g ${st.gColour}`;
    if (st.temperature !== 1) m += ` · T ${st.temperature}`;
    if (st.topP !== 1) m += ` · top-p ${st.topP}`;
    if (!mask && st.filter && res.stats.rejected) m += ` · filter redrew ${res.stats.rejected}`;
    m += ` · ${secs.toFixed(1)} s on ${state.ep === "webgpu" ? "WebGPU" : "CPU"}`;
    meta.textContent = m;
    sheetBtn.hidden = false;
    sheetBtn.onclick = () => {
      const a = document.createElement("a"); a.href = sheetUrl(all, w); a.download = `${slug(st.text)}_${st.size}_s${st.seed}_sheet.png`; a.click();
    };
  } catch (e) {
    if (frame) { cancelAnimationFrame(frame); frame = 0; }
    if (e.name === "AbortError") { meta.textContent = "stopped"; grid.querySelectorAll(".card").forEach((c) => c.remove()); el.remove(); }
    else if (state.ep === "webgpu" && state.model && /OrtRun|webgpu|map ?async|device|GPU/i.test(String(e.message || e))) {
      console.error(e);
      const why = String(e.message || e).replace(/^.*?(map async|device|lost|out of memory)/i, "$1").slice(0, 120);
      meta.className = "meta error";
      meta.textContent = `WebGPU failed on this device (${why}). Switching to the CPU…`;
      try {
        await switchToCPU(why);
        meta.textContent = "WebGPU failed on this device, so the page switched to the CPU. Press Draw again.";
      } catch (e2) {
        meta.textContent = `WebGPU failed and the CPU could not load: ${e2.message || e2}`;
      }
    } else { console.error(e); meta.className = "meta error"; meta.textContent = e.message || String(e); }
  } finally {
    markIdle();
    state.running = null;
    go.textContent = "Draw"; go.classList.remove("stop");
    if (!state.model) go.disabled = true;
    else status(readyText());
  }
}

$("form").addEventListener("submit", (e) => {
  e.preventDefault();
  if (state.running) { state.running.abort(); return; }
  if (!state.model) return;
  run(readSettings());
});

boot();
