// Mixing concepts without sliders: the words of the caption become concepts, and every sprite of the batch is
// drawn with its own variant of how they combine (which object is the main one, which layout).
// Prototype and checks: creative_auto.py, creative_regions.py, creative_attr.py in the project; lexicon:
// release/tools/build_lexicon.py.
//
// Rules (decided with the author, 15.11.2026):
// - concepts = words of the caption without stop words, known multi-word subjects kept together ("treasure chest"),
//   at most three; each is an object or an attribute (lexicon; a tap on the word chip flips it);
// - probability by position 1/2, 1/4, 1/8 ... normalised: in Latin script the LAST word is the strongest (English
//   puts the head noun last), in Cyrillic the first;
// - mixing only with two or more objects, or two or more attributes; otherwise the plain caption;
// - each sprite draws a variant uniformly from the enabled ones (chips on the page): three layouts with an order
//   (main object A ~ p, second object B among the rest ~ p): inside / over / centre; plus "blend" (both captions
//   everywhere, equal weights) and "caption" (the whole caption as one vector: the plain way, for comparison).
//   With two objects and everything on that is 2 + 2 + 2 + 1 + 1 = 8 variants. Zone weights stay at the checked
//   1.6 / 0.4: scaling them by p made the second object lose its own zone. A third object and every attribute act
//   on the whole sprite; attributes get strength ATTR * (0.5 + 0.5 p / max p).

export const MIX = { HI: 1.6, LO: 0.4, BLEND: 1.0, ATTR: 1.0, THIRD: 0.6, MAX: 3 };

// The variants, in the order of the chips. `ordered`: A and B play different parts (2 orders for two objects).
export const VARIANTS = {
  inside: { title: "inside", ordered: true, hint: "the first object in a middle band, the second above and below" },
  over: { title: "on top", ordered: true, hint: "the first object on top, the second below" },
  centre: { title: "centre", ordered: true, hint: "the first object in the centre, the second around it" },
  blend: { title: "blend", ordered: false, hint: "both objects everywhere, equal weights" },
  caption: { title: "one caption", ordered: false, hint: "the whole caption as one vector: the plain way" },
};
export const ALL_VARIANTS = Object.keys(VARIANTS);

/** Small SVG diagram of a variant: the first object in the accent colour, the second in grey. */
export function icon(kind) {
  const A = "var(--accent)", B = "var(--muted)";
  const body = {
    inside: `<rect width="12" height="4" fill="${B}"/><rect y="4" width="12" height="4" fill="${A}"/><rect y="8" width="12" height="4" fill="${B}"/>`,
    over: `<rect width="12" height="6" fill="${A}"/><rect y="6" width="12" height="6" fill="${B}"/>`,
    centre: `<rect width="12" height="12" fill="${B}"/><circle cx="6" cy="6" r="3.2" fill="${A}"/>`,
    blend: `<rect width="12" height="12" fill="${B}"/><path d="M0 12 L12 0 L12 4 L4 12 Z M0 4 L4 0 L8 0 L0 8 Z" fill="${A}"/>`,
    caption: `<rect x="0.5" y="1.5" width="11" height="9" rx="2" fill="none" stroke="${A}"/><path d="M3 5h6M3 7.5h4" stroke="${A}"/>`,
  }[kind];
  return `<svg class="vicon" viewBox="0 0 12 12" width="12" height="12" aria-hidden="true">${body}</svg>`;
}

const CYR = /[а-яёіїєґ]/i;
const WORD = /[\p{L}]+(?:['’][\p{L}]+)?/gu;
// Russian / Ukrainian adjectives by their ending. The multilingual encoder knows some of them badly ("жареный",
// "смажений" land next to "saucer"), so the ending decides before the nearest-neighbour guess.
const CYR_ADJ = /(ый|ий|ій|ой|ая|яя|ое|ее|ые|ие)$/i;
export const looksLikeAdjective = (w) => CYR.test(w) && CYR_ADJ.test(w);

export async function loadLexicon(base, fetchBytes) {
  const lx = JSON.parse(new TextDecoder().decode(await fetchBytes(`${base}/lexicon.json`)));
  const index = new Map(lx.entries.map(([t], i) => [t, i]));
  const kind = new Map(lx.entries);
  let vectors = null;
  return {
    stop: new Set(lx.stop), phrases: new Set(lx.phrases), kind, index, dim: lx.dim, entries: lx.entries,
    /** float32 vectors of all entries (2.8 MB, loaded on first use). */
    async vectors() {
      if (!vectors) {
        const b = await fetchBytes(`${base}/lexicon.f16`);
        vectors = halfRows(b, lx.dim);
      }
      return vectors;
    },
  };
}

function halfRows(bytes, dim) {
  const h = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength >> 1), out = new Float32Array(h.length);
  for (let i = 0; i < h.length; i++) {
    const v = h[i], s = v & 0x8000 ? -1 : 1, e = (v >> 10) & 31, f = v & 1023;
    out[i] = e === 0 ? s * f * 2 ** -24 : e === 31 ? (f ? NaN : s * Infinity) : s * (1 + f / 1024) * 2 ** (e - 15);
  }
  return { data: out, dim };
}

/** Caption -> concepts [{text, kind, known, p, rank}] (at most MIX.MAX) and whether the script is Cyrillic.
 * overrides: Map text -> kind, from taps on the word chips. */
export function parse(text, lex, overrides = new Map()) {
  const cyr = CYR.test(text);
  const words = (text.toLowerCase().match(WORD) || []).filter((w) => !lex.stop.has(w));
  const out = [];
  for (let i = 0; i < words.length;) {
    let n = 1;
    for (const k of [3, 2]) if (i + k <= words.length && lex.phrases.has(words.slice(i, i + k).join(" "))) { n = k; break; }
    out.push(words.slice(i, i + n).join(" "));
    i += n;
  }
  const extra = Math.max(0, out.length - MIX.MAX);
  const kept = out.slice(0, MIX.MAX);
  const ranks = kept.map((_, i) => (cyr ? i : kept.length - 1 - i));
  const raw = ranks.map((r) => 0.5 ** (r + 1)), sum = raw.reduce((a, b) => a + b, 0);
  const concepts = kept.map((t, i) => ({
    text: t, rank: ranks[i], p: raw[i] / sum,
    known: lex.kind.has(t) || looksLikeAdjective(t),
    kind: overrides.get(t) || lex.kind.get(t) || (looksLikeAdjective(t) ? "attribute" : "object"),
  }));
  return { concepts, cyr, dropped: extra };
}

/** Whether a variant fits the caption: one caption always; blend from two words; layouts need two objects.
 * Returns null when it fits, else the reason (for the disabled chip). */
export function unfit(kind, concepts) {
  const objects = concepts.filter((c) => c.kind === "object").length;
  if (kind === "caption") return null;
  if (kind === "blend") return concepts.length >= 2 ? null : "needs two words";
  return objects >= 2 ? null : "needs two objects, e.g. burger skull";
}

/** null = draw the plain caption (fewer than two words, or only "one caption" enabled); otherwise a plan whose
 * variants differ per sprite. enabled: variant kinds switched on; those that do not fit the caption are skipped. */
export function plan(concepts, enabled = ALL_VARIANTS) {
  if (concepts.length < 2) return null;
  const objects = concepts.filter((c) => c.kind === "object"), attributes = concepts.filter((c) => c.kind === "attribute");
  const kinds = ALL_VARIANTS.filter((k) => enabled.includes(k) && !unfit(k, concepts));
  if (!kinds.length || (kinds.length === 1 && kinds[0] === "caption")) return null;
  const pmax = Math.max(...concepts.map((c) => c.p)), n = objects.length;
  return {
    objects, attributes, concepts, kinds,
    count: kinds.reduce((s, k) => s + (VARIANTS[k].ordered ? n * (n - 1) : 1), 0),
    attrStrength: (c) => MIX.ATTR * (0.5 + 0.5 * c.p / pmax),
  };
}

function choose(items, rng) {
  const total = items.reduce((s, it) => s + it.p, 0);
  let u = rng() * total;
  for (const it of items) { u -= it.p; if (u <= 0) return it; }
  return items[items.length - 1];
}

/** One variant: {kind, a, b, others, attributes} (b = null without a second object). Unordered kinds keep the two
 * strongest objects as a and b, only for the label. */
export function drawVariant(pl, rng) {
  const kind = pl.kinds[Math.floor(rng() * pl.kinds.length)];
  if (pl.objects.length < 2) {  // one object (or none) and attributes: "blend" puts every word over the whole sprite
    const [a = null, ...others] = [...pl.objects];
    return { kind, a, b: null, others, attributes: pl.attributes };
  }
  let a, b;
  if (VARIANTS[kind].ordered) {
    a = choose(pl.objects, rng);
    b = choose(pl.objects.filter((o) => o !== a), rng);
  } else {
    [a, b] = [...pl.objects].sort((x, y) => y.p - x.p);
  }
  return { kind, a, b, others: pl.objects.filter((o) => o !== a && o !== b), attributes: pl.attributes };
}

/** Text under a sprite (next to icon(v.kind)): "skull / burger + fried" (A / B), "skull + burger" for a blend. */
export function label(v, caption = "") {
  if (v.kind === "caption") return `“${caption}”`;
  const head = v.b ? `${v.a.text} ${VARIANTS[v.kind].ordered ? "/" : "+"} ${v.b.text}` : v.a ? v.a.text : "";
  return [head, ...v.others.map((o) => o.text), ...v.attributes.map((t) => t.text)].filter(Boolean).join(" + ");
}

/** A sentence for the tooltip of a sprite. */
export function describe(v) {
  if (v.kind === "caption") return "the whole caption as one vector (the plain way)";
  if (!v.b) return "every word as its own caption over the whole sprite";
  if (v.kind === "inside") return `${v.a.text} in a middle band, ${v.b.text} above and below`;
  if (v.kind === "over") return `${v.a.text} on top, ${v.b.text} below`;
  if (v.kind === "centre") return `${v.a.text} in the centre, ${v.b.text} around`;
  if (v.kind === "blend") return `${v.a.text} and ${v.b.text} everywhere, equal weights`;
  return "the whole caption as one vector (the plain way)";
}

/** Layout value m(y, x) in [0, 1] (1 = A dominates) at window coordinates in [0, 1], with a one-pixel ramp. */
function layoutValue(layout, y, x, w) {
  const soft = (d) => Math.min(1, Math.max(0, d * w + 0.5));
  if (layout === "inside") return soft(Math.min(y - 0.35, 0.65 - y));
  if (layout === "over") return soft(0.5 - y);
  return soft(0.3 - Math.hypot(x - 0.5, y - 0.5));
}

/** The concepts of a variant (in order) and their weight maps over the S x S canvas; the layout is laid over the
 * visible w x w window at offset o (size 24 = a window on the 32 canvas). */
export function variantMaps(v, pl, S, o, w) {
  const concepts = [], maps = [];
  const fill = (val) => new Float32Array(S * S).fill(val);
  if (v.b && v.kind === "blend") {
    concepts.push(v.a, v.b); maps.push(fill(MIX.BLEND), fill(MIX.BLEND));
  } else if (v.b) {
    const ma = new Float32Array(S * S), mb = new Float32Array(S * S);
    for (let r = 0; r < S; r++) for (let c = 0; c < S; c++) {
      const m = layoutValue(v.kind, (r - o + 0.5) / w, (c - o + 0.5) / w, w);
      ma[r * S + c] = MIX.LO + (MIX.HI - MIX.LO) * m;
      mb[r * S + c] = MIX.HI + (MIX.LO - MIX.HI) * m;
    }
    concepts.push(v.a, v.b); maps.push(ma, mb);
  } else if (v.a) {
    concepts.push(v.a); maps.push(fill(1));
  }
  for (const c of v.others) { concepts.push(c); maps.push(fill(MIX.THIRD)); }
  for (const t of v.attributes) { concepts.push(t); maps.push(fill(pl.attrStrength(t))); }
  return { concepts, maps };
}

/** Vector of a concept: from the lexicon when it is there, else from the text encoder (loads it). */
export async function conceptVector(c, lex, encode) {
  if (lex.index.has(c.text)) {
    const V = await lex.vectors(), i = lex.index.get(c.text);
    return V.data.slice(i * V.dim, (i + 1) * V.dim);
  }
  return encode(c.text);
}

/** Kind of an unknown word from its nearest single-word lexicon entry (cosine of the encoder vectors; phrases are
 * skipped, so "fried" is not pulled towards "fried egg"). */
export async function nearestKind(vec, lex) {
  const V = await lex.vectors();
  let best = -2, bi = 0;
  for (let i = 0; i < lex.entries.length; i++) {
    if (lex.entries[i][0].includes(" ")) continue;
    let d = 0;
    for (let j = 0; j < V.dim; j++) d += V.data[i * V.dim + j] * vec[j];
    if (d > best) { best = d; bi = i; }
  }
  return { kind: lex.entries[bi][1], near: lex.entries[bi][0], cos: best };
}
