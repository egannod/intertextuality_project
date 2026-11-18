import { readPassageCSV } from "./csv.js";
import { PROMPT_TEMPLATE, toWords, makeWindows, commonPrefix, parseRowRefs, makeExperiment, makeFile, checkFile } from "./experiment.js";

// ------------------------------------------------------------------ settings you may want to change
// Where the notebook lives in your GitHub repository, for the "Open in molab" button. On a github.io address the
// owner and repository are worked out from the page's own address; set GITHUB_REPO ("owner/repository") to override.
const GITHUB_REPO = "";
const GITHUB_BRANCH = "main";
const NOTEBOOK_PATH = "notebook/intertextuality_molab.py";   // e.g. "docs/notebook/intertextuality_molab.py" if the site is in docs/

// Each example text is offered in Greek and in English (examples/<stem>_<greek|english>.csv).
const EXAMPLES = {
  query: [{ title: "Matthew 7:1-6", stem: "matthew_7_1-6" }, { title: "Matthew 7:1-13", stem: "matthew_7_1-13" }],
  corpus: [{ title: "1 Clement 13", stem: "1_clement_13" }, { title: "Psalm 25", stem: "psalm_25" }],
};

const $ = (s, root = document) => root.querySelector(s);
const REGION_INPUTS = ["#regions-quote", "#regions-allusion", "#regions-overlap"];
const $$ = (s, root = document) => [...root.querySelectorAll(s)];
const state = { query: null, corpus: null, list: [], titleEdited: false };

// Remembered between visits (if the browser allows it): the experiment list and the chosen models.
const store = {
  get(k) { try { return JSON.parse(localStorage.getItem(`itx2.${k}`) ?? "null"); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(`itx2.${k}`, JSON.stringify(v)); return true; } catch { return false; } },
};

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const plural = (n, one, many = `${one}s`) => `${n.toLocaleString()} ${n === 1 ? one : many}`;
const slug = (s) => (s || "experiment").normalize("NFKD").replace(/[:/\\.→]+/g, " ").replace(/[^\w\s-]/g, "").trim().replace(/[\s_]+/g, "-").toLowerCase().slice(0, 40) || "experiment";

// ------------------------------------------------------------------ passages
function makePassage(name, rows, notes = []) {
  const words = rows.flatMap((r) => toWords(r.text));
  const refs = rows.map((r) => r.ref);
  const prefix = commonPrefix(refs);
  return { name, rows, notes, refs, words, shortRefs: refs.map((r) => r.slice(prefix.length)) };
}

function guessLang(text) {
  const sample = text.slice(0, 400);
  const greek = (sample.match(/[Ͱ-Ͽἀ-῿]/g) || []).length;
  return greek > sample.length / 4 ? "grc" : "en";
}

function setPassage(kind, passage) {
  if (kind === "corpus" && state.corpus && state.corpus.name !== passage.name) {
    REGION_INPUTS.forEach((id) => { $(id).value = ""; });   // marked verses refer to the old corpus
  }
  state[kind] = passage;
  const panel = $(`#${kind}-panel`);
  const maxRows = 80;
  const body = passage.rows.slice(0, maxRows)
    .map((r, i) => `<span class="ref">${esc(passage.shortRefs[i] || r.ref)}</span>${esc(r.text)}`).join(" ");
  const more = passage.rows.length > maxRows ? ` <span class="ref">…</span>${plural(passage.rows.length - maxRows, "more row")}` : "";
  $(".passage-preview", panel).innerHTML =
    `<p class="passage-meta"><strong>${esc(passage.name)}</strong>, ${plural(passage.rows.length, "row")}, ${plural(passage.words.length, "word")}</p>` +
    `<p class="passage-text" lang="${guessLang(passage.words.join(" "))}">${body}${more}</p>` +
    passage.notes.map((n) => `<p class="passage-note">${esc(n)}</p>`).join("");
  $$(".examples button", panel).forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.name === passage.name)));
  if (!state.titleEdited) $("#exp-title").value = autoTitle();
  updateForm();
}

function autoTitle() {
  const { query, corpus } = state;
  if (query && corpus) return `${query.name} → ${corpus.name}`;
  return "";
}

async function loadFile(kind, file) {
  clearError();
  try {
    const { rows, notes } = readPassageCSV(await file.text());
    setPassage(kind, makePassage(file.name.replace(/\.(csv|tsv|txt)$/i, "").replace(/_+/g, " ").trim(), rows, notes));
  } catch (e) {
    showError(`<strong>Could not read ${esc(file.name)}.</strong> ${esc(e.message || e)}`);
  }
}

async function loadExample(kind, ex) {
  clearError();
  try {
    const res = await fetch(`examples/${ex.file}`);
    if (!res.ok) throw new Error(`the example file returned ${res.status}`);
    const { rows, notes } = readPassageCSV(await res.text());
    setPassage(kind, makePassage(ex.name, rows, notes));
  } catch (e) {
    showError(`<strong>Could not load the example.</strong> ${esc(e.message || e)}`);
  }
}

function setupPanel(kind) {
  const panel = $(`#${kind}-panel`);
  const input = $(".file-input", panel), drop = $(".drop", panel);
  input.addEventListener("change", () => { if (input.files[0]) loadFile(kind, input.files[0]); input.value = ""; });
  for (const ev of ["dragenter", "dragover"]) drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("dragging"); });
  for (const ev of ["dragleave", "drop"]) drop.addEventListener(ev, () => drop.classList.remove("dragging"));
  drop.addEventListener("drop", (e) => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) loadFile(kind, f); });
  const holder = $(".example-buttons", panel);
  EXAMPLES[kind].forEach((ex, k) => {
    const group = document.createElement("span");
    group.className = "example-group";
    group.append(`${ex.title} in `);
    [["Greek", "greek"], ["English", "english"]].forEach(([lang, suffix], j) => {
      const b = document.createElement("button");
      const name = `${ex.title} (${lang})`;
      b.type = "button"; b.textContent = lang; b.dataset.name = name; b.setAttribute("aria-pressed", "false");
      b.setAttribute("aria-label", `${ex.title} in ${lang}`);
      b.addEventListener("click", () => loadExample(kind, { file: `${ex.stem}_${suffix}.csv`, name }));
      group.append(b);
      if (j === 0) group.append(" or ");
    });
    if (k < EXAMPLES[kind].length - 1) group.append(";");
    holder.append(group);
  });
}

// ------------------------------------------------------------------ the experiment form
function windowSettings() {
  const size = Math.max(2, Math.min(400, parseInt($("#window-size").value, 10) || 40));
  const stride = Math.max(1, Math.min(400, parseInt($("#stride").value, 10) || 4));
  return { size, stride };
}

function currentRegions() {
  if (!state.corpus) return { regions: [], unknown: [], quote: [], allusion: [], overlap: [] };
  const q = parseRowRefs($("#regions-quote").value, state.corpus.refs);
  const al = parseRowRefs($("#regions-allusion").value, state.corpus.refs);
  const o = parseRowRefs($("#regions-overlap").value, state.corpus.refs);
  const regions = [
    ...q.ranges.map(([from, to]) => ({ from, to, kind: "quotation", label: "" })),
    ...al.ranges.map(([from, to]) => ({ from, to, kind: "allusion", label: "" })),
    ...o.ranges.map(([from, to]) => ({ from, to, kind: "overlap", label: "" })),
  ];
  return { regions, unknown: [...q.unknown, ...al.unknown, ...o.unknown], quote: q.ranges, allusion: al.ranges, overlap: o.ranges };
}

const showRange = ([a, b]) => (a === b ? a : `${a} to ${b}`);

function updateForm() {
  const { size, stride } = windowSettings();
  const wc = $("#window-count");
  if (state.corpus) {
    const n = makeWindows(state.corpus.words.length, size, stride).length;
    wc.textContent = `${plural(n, "window")} over ${plural(state.corpus.words.length, "word")}`;
  } else wc.textContent = "";

  const check = $("#region-check");
  const typed = REGION_INPUTS.some((id) => $(id).value.trim());
  if (!typed) check.textContent = "";
  else if (!state.corpus) check.textContent = "Add a corpus to check these references.";
  else {
    const r = currentRegions();
    const parts = [];
    if (r.quote.length) parts.push(`Quoting: ${r.quote.map(showRange).join(", ")}.`);
    if (r.allusion.length) parts.push(`Proposed allusion: ${r.allusion.map(showRange).join(", ")}.`);
    if (r.overlap.length) parts.push(`Shared words: ${r.overlap.map(showRange).join(", ")}.`);
    if (r.unknown.length) parts.push(`Not found in the corpus: ${r.unknown.join(", ")}.`);
    check.textContent = parts.join(" ");
    check.classList.toggle("warn", r.unknown.length > 0);
  }

  const ready = !!(state.query && state.corpus);
  $("#add-button").disabled = !ready;
  $("#ready-hint").textContent = ready ? "" : !state.query && !state.corpus ? "Add a query and a corpus first."
    : !state.query ? "Add a query first." : "Add a corpus first.";
}

function uniqueId(base) {
  const taken = new Set(state.list.map((e) => e.id));
  let id = slug(base), k = 2;
  while (taken.has(id)) id = `${slug(base)}-${k++}`;
  return id;
}

function addExperiment() {
  clearError();
  const { query, corpus } = state;
  if (!query || !corpus) return;
  const { size, stride } = windowSettings();
  const title = $("#exp-title").value.trim() || autoTitle();
  const exp = makeExperiment({
    id: uniqueId(title), title, kind: $("#exp-kind").value, query, corpus,
    windowWords: Math.min(size, corpus.words.length), strideWords: stride, regions: currentRegions().regions,
  });
  state.list.push(exp);
  saveList();
  renderList();
  state.titleEdited = false;
  // what you expect and the marked verses belong to this pair of texts, so start the next one afresh
  $("#exp-kind").value = "";
  REGION_INPUTS.forEach((id) => { $(id).value = ""; });
  updateForm();
  $("#ready-hint").textContent = `Added “${title}”. Choose other texts to add another experiment.`;
}

// ------------------------------------------------------------------ the list and the file
function saveList() {
  if (!store.set("list", state.list)) {
    $("#file-summary").textContent = "This list is too large to remember in the browser; download the file before leaving the page.";
  }
}

function selectedModels() {
  const models = $$("input[name=model]:checked").map((c) => c.value);
  const other = $("#other-model").value.trim();
  if (other && !models.includes(other)) models.push(other);
  return models;
}

function saveModels() {
  store.set("models", { checked: $$("input[name=model]:checked").map((c) => c.value), other: $("#other-model").value.trim() });
}

function renderList() {
  const holder = $("#exp-list");
  if (!state.list.length) {
    holder.innerHTML = `<p class="empty">No experiments yet. Add the one above, or start from the four built-in experiments.</p>`;
  } else {
    holder.innerHTML = `<ol>${state.list.map((e, i) => {
      const qWords = e.query.rows.reduce((n, r) => n + toWords(r.text).length, 0);
      const cWords = e.corpus.rows.reduce((n, r) => n + toWords(r.text).length, 0);
      const nWin = makeWindows(cWords, e.window_words, e.stride_words).length;
      const marks = (e.regions || []).length ? ` · ${plural(e.regions.length, "marked region")}` : "";
      return `<li>
        <div class="exp-main">
          <p class="exp-title">${esc(e.title)}${e.kind ? ` <span class="badge">${esc(e.kind)}</span>` : ""}</p>
          <p class="exp-meta">Query ${esc(e.query.name)} (${plural(qWords, "word")}) · corpus ${esc(e.corpus.name)} (${plural(cWords, "word")}) · window ${e.window_words}, step ${e.stride_words} · ${plural(nWin, "window")}${marks}</p>
        </div>
        <button type="button" class="remove" data-index="${i}" aria-label="Remove ${esc(e.title)}">Remove</button>
      </li>`;
    }).join("")}</ol>`;
  }
  updateFileSummary();
}

function updateFileSummary() {
  const models = selectedModels();
  const windows = state.list.reduce((n, e) => {
    const words = e.corpus.rows.reduce((m, r) => m + toWords(r.text).length, 0);
    return n + makeWindows(words, e.window_words, e.stride_words).length;
  }, 0);
  const ok = state.list.length > 0 && models.length > 0;
  $("#download-button").disabled = !ok;
  $("#file-summary").textContent = !state.list.length ? "Add at least one experiment."
    : !models.length ? "Choose at least one model."
    : `${plural(state.list.length, "experiment")} · ${plural(models.length, "model")} · ${plural(windows * models.length, "window")} to score in all`;
}

function fileName(file) {
  const base = state.list.length === 1 ? slug(state.list[0].title) : "intertextuality-experiments";
  return `${base}.json`;
}

function downloadFile() {
  clearError();
  const file = makeFile({ models: selectedModels(), experiments: state.list });
  const problems = checkFile(file);
  if (problems.length) {
    showError(`<strong>The file would not work in the notebook.</strong> ${problems.map(esc).join(" ")}`);
    return;
  }
  const blob = new Blob([JSON.stringify(file, null, 1)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = fileName(file);
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  $("#file-summary").textContent = `Saved ${a.download}. Now open the notebook in molab (step 4).`;
}

function addFromFile(obj, sourceName) {
  const problems = checkFile(obj);
  if (problems.length) throw new Error(problems.join(" "));
  for (const e of obj.experiments) {
    state.list.push({ ...e, id: uniqueId(e.id || e.title) });
  }
  if (Array.isArray(obj.models) && obj.models.length) {
    const known = $$("input[name=model]").map((c) => c.value);
    $$("input[name=model]").forEach((c) => { c.checked = obj.models.includes(c.value); });
    const other = obj.models.find((m) => !known.includes(m));
    if (other) $("#other-model").value = other;
    saveModels();
  }
  saveList();
  renderList();
  $("#file-summary").textContent = `Added ${plural(obj.experiments.length, "experiment")} from ${sourceName}.`;
}

async function addExampleFile(path, what) {
  clearError();
  try {
    const res = await fetch(path);
    if (!res.ok) throw new Error(`the file returned ${res.status}`);
    addFromFile(await res.json(), what);
  } catch (e) {
    showError(`<strong>Could not load ${esc(what)}.</strong> ${esc(e.message || e)}`);
  }
}

async function openFile(file) {
  clearError();
  try {
    addFromFile(JSON.parse(await file.text()), file.name);
  } catch (e) {
    showError(`<strong>Could not use ${esc(file.name)}.</strong> ${esc(e.message || e)}`);
  }
}

let clearArmed = null;
function clearList(button) {
  if (!state.list.length) return;
  if (!clearArmed) {
    button.textContent = "Click again to clear all";
    clearArmed = setTimeout(() => { clearArmed = null; button.textContent = "Clear the list"; }, 4000);
    return;
  }
  clearTimeout(clearArmed); clearArmed = null;
  button.textContent = "Clear the list";
  state.list = [];
  saveList();
  renderList();
}

// ------------------------------------------------------------------ molab link
function githubRepo() {
  if (GITHUB_REPO) return GITHUB_REPO;
  const m = location.hostname.match(/^([^.]+)\.github\.io$/i);
  if (!m) return null;
  const first = location.pathname.split("/").filter(Boolean)[0];
  return first && !first.includes(".") ? `${m[1]}/${first}` : `${m[1]}/${m[1]}.github.io`;
}

function setupMolabLink() {
  const repo = githubRepo();
  const link = $("#molab-link"), note = $("#molab-note");
  if (repo) {
    link.href = `https://molab.marimo.io/github/${repo}/blob/${GITHUB_BRANCH}/${NOTEBOOK_PATH}`;
    note.textContent = `opens ${NOTEBOOK_PATH} from ${repo}`;
  } else {
    link.href = "https://molab.marimo.io";
    note.textContent = "Once this site is on GitHub Pages, this button opens the notebook straight from your repository. For now it opens molab’s home page.";
  }
}

// ------------------------------------------------------------------ errors
function showError(html) { const e = $("#error"); e.innerHTML = html; e.hidden = false; }
function clearError() { $("#error").hidden = true; $("#error").innerHTML = ""; }

// ------------------------------------------------------------------ start
function init() {
  setupPanel("query");
  setupPanel("corpus");
  $("#prompt-text").textContent = PROMPT_TEMPLATE;
  setupMolabLink();

  for (const id of ["#window-size", "#stride", ...REGION_INPUTS]) $(id).addEventListener("input", updateForm);
  $("#exp-title").addEventListener("input", () => { state.titleEdited = $("#exp-title").value.trim() !== ""; });
  $("#add-button").addEventListener("click", addExperiment);
  $("#download-button").addEventListener("click", downloadFile);
  $("#add-defaults").addEventListener("click", () => addExampleFile("examples/default_experiments.json", "the built-in experiments"));
  $("#add-suggested").addEventListener("click", () => addExampleFile("examples/suggested_experiments.json", "the suggested experiments"));
  $("#clear-list").addEventListener("click", (e) => clearList(e.currentTarget));
  $("#open-file").addEventListener("change", (e) => { const f = e.target.files[0]; if (f) openFile(f); e.target.value = ""; });
  $("#exp-list").addEventListener("click", (e) => {
    const b = e.target.closest("button.remove");
    if (!b) return;
    state.list.splice(Number(b.dataset.index), 1);
    saveList();
    renderList();
  });
  $$("input[name=model]").forEach((c) => c.addEventListener("change", () => { saveModels(); updateFileSummary(); }));
  $("#other-model").addEventListener("input", () => { saveModels(); updateFileSummary(); });

  const savedModels = store.get("models");
  if (savedModels) {
    $$("input[name=model]").forEach((c) => { c.checked = savedModels.checked.includes(c.value); });
    $("#other-model").value = savedModels.other || "";
  }
  const savedList = store.get("list");
  if (Array.isArray(savedList)) state.list = savedList.filter((e) => e && e.query && e.corpus);
  renderList();
  updateForm();
}

init();
