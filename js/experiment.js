// The experiment file: what the molab notebook reads. No dependencies, so it can be tested in Node.

export const FORMAT = "sliding-window-intertextuality";
export const VERSION = 1;

// Shown on the page for reference; the notebook holds the copy it actually uses.
export const PROMPT_TEMPLATE = `Compare the two passages below.

QUERY:
{query}

EXCERPT:
{excerpt}

The EXCERPT is a short run of words cut from a longer text, so it may begin or end in the middle of a sentence. How closely related are the EXCERPT and the QUERY? Look for quotation or close paraphrase, shared distinctive words and phrases (especially rare ones), and distinctive shared images or ideas. Common words and generic religious language count for little.

0 = unrelated
1-3 = only superficial or generic overlap
4-6 = a probable allusion or echo: distinctive shared words, phrases or ideas
7-8 = a strong allusion with several distinctive shared words or phrases
9-10 = one passage quotes or closely paraphrases the other

Answer with a single integer from 0 to 10 and nothing else.`;

/** Split text into words exactly as the notebook does (Unicode NFC, any run of whitespace). */
export function toWords(text) {
  const t = (text || "").normalize("NFC").trim();
  return t ? t.split(/\s+/u) : [];
}

/** Window boundaries over n words: the same rule as the notebook's make_windows. */
export function makeWindows(n, size, stride) {
  if (n === 0) return [];
  size = Math.max(1, Math.min(size, n));
  stride = Math.max(1, stride);
  const starts = [];
  for (let s = 0; s + size <= n; s += stride) starts.push(s);
  if (starts[starts.length - 1] + size < n) starts.push(n - size);
  return starts.map((s, i) => ({ window: i + 1, start: s, end: s + size }));
}

/** Longest shared beginning of all labels, ending at a space, colon or full stop ("Psalm 25:"). */
export function commonPrefix(labels) {
  if (labels.length < 2) return "";
  let p = labels[0];
  for (const l of labels) { while (!l.startsWith(p)) p = p.slice(0, -1); }
  const cut = Math.max(p.lastIndexOf(" "), p.lastIndexOf(":"), p.lastIndexOf("."));
  return cut >= 0 && labels.every((l) => l.length > cut + 1) ? p.slice(0, cut + 1) : "";
}

/**
 * Read a list of references typed by the visitor, such as "13:2" or "25:15-16; 25:18", against the corpus rows.
 * Each item is a reference as it appears in the CSV (in full, or without the shared beginning such as "Psalm 25:"),
 * or two such references joined by a dash. Returns {ranges: [[fromRef, toRef]], unknown: [items not found]}.
 */
export function parseRowRefs(input, refs) {
  const prefix = commonPrefix(refs);
  const find = (s) => {
    s = s.trim().normalize("NFC");
    if (!s) return -1;
    let i = refs.indexOf(s);
    if (i < 0 && prefix) i = refs.indexOf(prefix + s);
    if (i < 0) i = refs.findIndex((r) => r.toLowerCase() === s.toLowerCase());
    // "25:9" for "Psalm 25:9": the typed text is the end of exactly one reference, after a space or colon
    if (i < 0) {
      const hits = refs.map((r, k) => [r, k]).filter(([r]) => r.endsWith(" " + s) || r.endsWith(":" + s));
      if (hits.length === 1) i = hits[0][1];
    }
    return i;
  };
  const ranges = [], unknown = [];
  for (const raw of (input || "").split(/[;,\n]+/)) {
    const item = raw.trim();
    if (!item) continue;
    let a = find(item), b = a;
    if (a < 0) {
      const parts = item.split(/\s*[-–—]\s*/);
      if (parts.length === 2) {
        a = find(parts[0]);
        // "15-16" after "Psalm 25:15": the second part may leave out what the first part has
        b = find(parts[1]);
        if (b < 0 && a >= 0) {
          const head = refs[a].slice(0, Math.max(refs[a].lastIndexOf(":"), refs[a].lastIndexOf(" ")) + 1);
          b = find(head + parts[1]);
        }
      }
    }
    if (a < 0 || b < 0) { unknown.push(item); continue; }
    ranges.push(a <= b ? [refs[a], refs[b]] : [refs[b], refs[a]]);
  }
  return { ranges, unknown };
}

/** One experiment as stored in the file. */
export function makeExperiment({ id, title, kind, query, corpus, windowWords, strideWords, regions, notes }) {
  const e = {
    id,
    title,
    query: { name: query.name, rows: query.rows.map((r) => ({ reference: r.ref, text: r.text })) },
    corpus: { name: corpus.name, rows: corpus.rows.map((r) => ({ reference: r.ref, text: r.text })) },
    window_words: windowWords,
    stride_words: strideWords,
  };
  if (kind) e.kind = kind;
  if (regions && regions.length) e.regions = regions;
  if (notes) e.notes = notes;
  return e;
}

/** The whole file. */
export function makeFile({ title, models, experiments }) {
  return {
    format: FORMAT,
    version: VERSION,
    title: title || experiments.map((e) => e.title).join("; "),
    created: new Date().toISOString(),
    created_with: "website",
    models,
    experiments,
  };
}

/** Check a parsed file the way the notebook will, so a bad file is caught before it reaches molab. */
export function checkFile(obj) {
  const problems = [];
  if (!obj || typeof obj !== "object" || !Array.isArray(obj.experiments) || !obj.experiments.length) {
    return ["The file has no list of experiments."];
  }
  obj.experiments.forEach((e, i) => {
    const where = `Experiment ${i + 1}`;
    for (const part of ["query", "corpus"]) {
      const rows = e?.[part]?.rows;
      if (!Array.isArray(rows) || !rows.some((r) => toWords(r?.text).length)) problems.push(`${where}: the ${part} has no text.`);
    }
    if (!Number.isInteger(e.window_words) || e.window_words < 2) problems.push(`${where}: the window must be a whole number of at least 2 words.`);
    if (!Number.isInteger(e.stride_words) || e.stride_words < 1) problems.push(`${where}: the step must be a whole number of at least 1 word.`);
    const refs = (e?.corpus?.rows || []).map((r) => r.reference);
    for (const g of e.regions || []) {
      if (!refs.includes(g.from) || !refs.includes(g.to)) problems.push(`${where}: a marked region (${g.from}–${g.to}) does not match the corpus references.`);
    }
  });
  return problems;
}
