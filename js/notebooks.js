// Fill the notebook templates with the visitor's texts, experiments and models. No DOM, so it can be tested in Node.

const PLACEHOLDER = '"__DATA__"';

/** A JavaScript string as a Python string literal (JSON string syntax is valid Python, bar two line separators). */
export function pythonString(s) {
  const LS = String.fromCharCode(0x2028), PS = String.fromCharCode(0x2029);
  return JSON.stringify(s).split(LS).join("\\u2028").split(PS).join("\\u2029");
}

/**
 * The data both notebooks read:
 * {format, version, created, models: [ids], texts: {id: {name, rows: [{reference, text}]}}, experiments: [{query, corpus}]}
 * Only texts used by an experiment are included.
 */
export function buildData({ queries, corpora, experiments, models }) {
  const texts = {};
  const add = (t) => { texts[t.id] = { name: t.name, rows: t.rows.map((r) => ({ reference: r.ref, text: r.text })) }; };
  const byId = new Map([...queries, ...corpora].map((t) => [t.id, t]));
  const used = [];
  for (const e of experiments) {
    const q = byId.get(e.query), c = byId.get(e.corpus);
    if (!q || !c) continue;
    add(q); add(c);
    used.push({ query: q.id, corpus: c.id });
  }
  return { format: "sliding-window-intertextuality", version: 2, created: new Date().toISOString(), models, texts, experiments: used };
}

export function fillColab(templateText, data) {
  const nb = JSON.parse(templateText);
  const literal = pythonString(JSON.stringify(data));
  let filled = 0;
  for (const cell of nb.cells) {
    const src = Array.isArray(cell.source) ? cell.source.join("") : cell.source;
    if (src.includes(PLACEHOLDER)) {
      cell.source = src.replace(PLACEHOLDER, () => literal);
      filled++;
    }
  }
  if (filled !== 1) throw new Error("The Colab template is damaged (its data placeholder was not found).");
  return JSON.stringify(nb, null, 1) + "\n";
}

export function fillMolab(templateText, data) {
  if (templateText.split(PLACEHOLDER).length !== 2) throw new Error("The molab template is damaged (its data placeholder was not found).");
  return templateText.replace(PLACEHOLDER, () => pythonString(JSON.stringify(data)));
}
