import { readPassageCSV } from "./csv.js";
import { buildData, fillColab, fillMolab } from "./notebooks.js";

const $ = (s, root = document) => root.querySelector(s);
const $$ = (s, root = document) => [...root.querySelectorAll(s)];
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const plural = (n, one, many = `${one}s`) => `${n.toLocaleString()} ${n === 1 ? one : many}`;
const wordCount = (rows) => rows.reduce((n, r) => n + (r.text.normalize("NFC").trim().split(/\s+/u).filter(Boolean).length), 0);

const state = { query: [], corpus: [], experiments: [], nextId: 1 };

// Remembered between visits when the browser allows it.
const store = {
  get(k) { try { return JSON.parse(localStorage.getItem(`itxnb.${k}`) ?? "null"); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(`itxnb.${k}`, JSON.stringify(v)); } catch { /* full or blocked */ } },
};
const save = () => store.set("state", state);

// ------------------------------------------------------------------ texts
function uniqueName(kind, name) {
  const taken = new Set(state[kind].map((t) => t.name));
  let n = name, k = 2;
  while (taken.has(n)) n = `${name} (${k++})`;
  return n;
}

async function addFiles(kind, files) {
  clearError();
  const problems = [];
  for (const file of files) {
    try {
      const { rows } = readPassageCSV(await file.text());
      if (!rows.length) throw new Error("it has no text");
      const name = uniqueName(kind, file.name.replace(/\.(csv|tsv|txt)$/i, "").replace(/_+/g, " ").trim() || file.name);
      state[kind].push({ id: `${kind === "query" ? "q" : "c"}${state.nextId++}`, name, rows });
    } catch (e) {
      problems.push(`${esc(file.name)}: ${esc(e.message || e)}`);
    }
  }
  if (problems.length) showError(`<strong>Could not read</strong> ${problems.join("; ")}`);
  if (!state.experiments.length && state.query.length && state.corpus.length) {
    state.experiments.push({ query: state.query[0].id, corpus: state.corpus[0].id });
  }
  save();
  render();
}

function removeText(kind, id) {
  state[kind] = state[kind].filter((t) => t.id !== id);
  const key = kind === "query" ? "query" : "corpus";
  for (const e of state.experiments) if (e[key] === id) e[key] = "";
  save();
  render();
}

function renderTexts(kind) {
  const list = $(`.column[data-kind="${kind}"] .text-list`);
  list.innerHTML = state[kind].map((t) => `
    <li>
      <span class="text-name">${esc(t.name)}</span>
      <span class="text-meta">${plural(t.rows.length, "row")} · ${plural(wordCount(t.rows), "word")}</span>
      <button type="button" class="remove" data-kind="${kind}" data-id="${t.id}" aria-label="Remove ${esc(t.name)}">×</button>
    </li>`).join("");
}

function setupDrop(kind) {
  const column = $(`.column[data-kind="${kind}"]`);
  const input = $(".file-input", column), drop = $(".drop", column);
  input.addEventListener("change", () => { addFiles(kind, [...input.files]); input.value = ""; });
  for (const ev of ["dragenter", "dragover"]) drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("dragging"); });
  for (const ev of ["dragleave", "drop"]) drop.addEventListener(ev, () => drop.classList.remove("dragging"));
  drop.addEventListener("drop", (e) => { e.preventDefault(); addFiles(kind, [...e.dataTransfer.files]); });
  $(".text-list", column).addEventListener("click", (e) => {
    const b = e.target.closest("button.remove");
    if (b) removeText(b.dataset.kind, b.dataset.id);
  });
}

// ------------------------------------------------------------------ experiments
function options(kind, selected) {
  const blank = `<option value=""${selected ? "" : " selected"} disabled>Choose a ${kind === "query" ? "query" : "corpus"}</option>`;
  return blank + state[kind].map((t) => `<option value="${t.id}"${t.id === selected ? " selected" : ""}>${esc(t.name)}</option>`).join("");
}

function renderExperiments() {
  $("#experiments").innerHTML = state.experiments.map((e, i) => `
    <li>
      <select data-index="${i}" data-key="query" aria-label="Query of experiment ${i + 1}" class="${e.query ? "" : "missing"}">${options("query", e.query)}</select>
      <span class="bar" aria-hidden="true">|</span>
      <select data-index="${i}" data-key="corpus" aria-label="Corpus of experiment ${i + 1}" class="${e.corpus ? "" : "missing"}">${options("corpus", e.corpus)}</select>
      <button type="button" class="remove" data-index="${i}" aria-label="Remove experiment ${i + 1}">×</button>
    </li>`).join("");
  $("#add-experiment").disabled = !(state.query.length && state.corpus.length);
}

// ------------------------------------------------------------------ models and download
function selectedModels() {
  const models = $$("input[name=model]:checked").map((c) => c.value);
  const other = $("#other-model").value.trim().replace(/^https?:\/\/huggingface\.co\//i, "").replace(/\/+$/, "");
  if ($("#other-on").checked && other && !models.includes(other)) models.push(other);
  return models;
}

function readyExperiments() {
  return state.experiments.filter((e) => e.query && e.corpus);
}

function renderStatus() {
  const exps = readyExperiments(), models = selectedModels();
  const ok = exps.length > 0 && models.length > 0;
  $("#download-colab").disabled = $("#download-molab").disabled = !ok;
  $("#status").textContent = !state.query.length || !state.corpus.length ? "Add at least one query and one corpus."
    : !exps.length ? "Choose a query and a corpus for at least one experiment."
    : !models.length ? "Choose at least one model."
    : `${plural(exps.length, "experiment")} · ${plural(models.length, "model")}`;
}

function render() {
  renderTexts("query");
  renderTexts("corpus");
  renderExperiments();
  renderStatus();
}

async function download(kind) {
  clearError();
  const data = buildData({ queries: state.query, corpora: state.corpus, experiments: readyExperiments(), models: selectedModels() });
  try {
    const res = await fetch(kind === "colab" ? "templates/colab.ipynb" : "templates/molab.py", { cache: "no-cache" });
    if (!res.ok) throw new Error(`the template returned ${res.status}`);
    const template = await res.text();
    const text = kind === "colab" ? fillColab(template, data) : fillMolab(template, data);
    const name = kind === "colab" ? "intertextuality_colab.ipynb" : "intertextuality_molab.py";
    const blob = new Blob([text], { type: kind === "colab" ? "application/x-ipynb+json" : "text/x-python" });
    const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: name });
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  } catch (e) {
    showError(`<strong>Could not make the notebook.</strong> ${esc(e.message || e)}`);
  }
}

function showError(html) { const e = $("#error"); e.innerHTML = html; e.hidden = false; }
function clearError() { $("#error").hidden = true; }

// ------------------------------------------------------------------ start
function init() {
  const saved = store.get("state");
  if (saved && Array.isArray(saved.query) && Array.isArray(saved.corpus) && Array.isArray(saved.experiments)) {
    Object.assign(state, saved);
  }
  const models = store.get("models");
  if (models) {
    $$("input[name=model]").forEach((c) => { c.checked = models.checked.includes(c.value); });
    $("#other-on").checked = !!models.otherOn;
    $("#other-model").value = models.other || "";
  }
  setupDrop("query");
  setupDrop("corpus");
  $("#add-experiment").addEventListener("click", () => {
    state.experiments.push({ query: state.query[0]?.id || "", corpus: state.corpus[0]?.id || "" });
    save();
    render();
  });
  $("#experiments").addEventListener("change", (e) => {
    const s = e.target.closest("select");
    if (!s) return;
    state.experiments[Number(s.dataset.index)][s.dataset.key] = s.value;
    save();
    render();
  });
  $("#experiments").addEventListener("click", (e) => {
    const b = e.target.closest("button.remove");
    if (!b) return;
    state.experiments.splice(Number(b.dataset.index), 1);
    save();
    render();
  });
  const saveModels = () => {
    store.set("models", { checked: $$("input[name=model]:checked").map((c) => c.value), otherOn: $("#other-on").checked, other: $("#other-model").value });
    renderStatus();
  };
  $$("input[name=model], #other-on").forEach((c) => c.addEventListener("change", saveModels));
  $("#other-model").addEventListener("input", () => {
    $("#other-on").checked = $("#other-model").value.trim() !== "";
    saveModels();
  });
  $("#download-colab").addEventListener("click", () => download("colab"));
  $("#download-molab").addEventListener("click", () => download("molab"));
  render();
}

init();
