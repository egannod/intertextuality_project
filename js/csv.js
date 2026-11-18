// CSV reading for query and corpus files. RFC 4180: quoted fields, "" escapes, commas and newlines inside quotes,
// CRLF or LF line ends, optional UTF-8 byte-order mark. Semicolon- and tab-separated files are detected too.

function parseWith(text, sep) {
  const rows = [];
  let row = [], field = "", i = 0, inQuotes = false;
  while (i < text.length) {
    const ch = text[i];
    if (inQuotes) {
      if (ch === '"') {
        if (text[i + 1] === '"') { field += '"'; i += 2; continue; }
        inQuotes = false; i++; continue;
      }
      field += ch; i++; continue;
    }
    if (ch === '"' && field.trim() === "") { inQuotes = true; field = ""; i++; continue; }
    if (ch === sep) { row.push(field); field = ""; i++; continue; }
    if (ch === "\r" || ch === "\n") {
      row.push(field); rows.push(row); row = []; field = "";
      i += ch === "\r" && text[i + 1] === "\n" ? 2 : 1;
      continue;
    }
    field += ch; i++;
  }
  if (inQuotes) throw new Error("The file ends inside a quoted field (an opening \" without a closing \").");
  if (field !== "" || row.length) { row.push(field); rows.push(row); }
  return rows.filter((r) => r.some((c) => c.trim() !== ""));
}

/**
 * Parse CSV text. The separator (comma, semicolon or tab) is the one that gives every row the same
 * number of columns; if none does, each line is read as a single field.
 */
export function parseCSV(text) {
  text = text.replace(/^﻿/, "");
  let best = null;
  for (const sep of [",", "\t", ";"]) {
    let rows;
    try { rows = parseWith(text, sep); } catch { continue; }
    const widths = new Set(rows.map((r) => r.length));
    const width = rows.length ? rows[0].length : 0;
    if (widths.size === 1 && width > 1 && (!best || width > best.width)) best = { sep, rows, width };
  }
  if (best) return { rows: best.rows, sep: best.sep, singleColumn: false };
  return { rows: parseWith(text, "\u0000"), sep: null, singleColumn: true };
}

const TEXT_NAMES = ["text", "passage", "content", "verse_text", "body"];
const REF_NAMES = ["reference", "ref", "verse", "label", "id", "section", "citation"];

/**
 * Turn CSV text into passage rows [{ref, text}].
 * Header row: a column named text (or passage/content) holds the text; reference (or ref/verse/label) labels it.
 * No header: one column = text; two or more columns = first is the reference, last is the text.
 */
export function readPassageCSV(csvText) {
  const parsed = parseCSV(csvText);
  const rows = parsed.rows;
  if (!rows.length) throw new Error("The file is empty.");
  const head = rows[0].map((c) => c.trim().toLowerCase());
  let textCol = head.findIndex((h) => TEXT_NAMES.includes(h));
  let refCol = head.findIndex((h) => REF_NAMES.includes(h));
  let body = rows, notes = [];
  if (parsed.singleColumn && rows.length > 1 && rows.some((r) => /[,;\t]/.test(r[0]))) {
    notes.push("The rows have different numbers of separators, so each line is read as one piece of text. Put quotes around text that contains commas.");
  }
  if (textCol >= 0) {
    body = rows.slice(1);
  } else {
    const width = Math.max(...rows.map((r) => r.length));
    textCol = width - 1;
    refCol = width > 1 ? 0 : -1;
    notes.push(width > 1
      ? "No header row found: reading the first column as the reference and the last as the text."
      : "No header row found: reading each row as text.");
  }
  const out = [];
  body.forEach((r, k) => {
    const text = (r[textCol] ?? "").replace(/\s+/gu, " ").trim().normalize("NFC");
    if (!text) return;
    const ref = refCol >= 0 ? (r[refCol] ?? "").trim() : "";
    out.push({ ref: ref || `row ${k + 1}`, text });
  });
  if (!out.length) throw new Error("No text found. Put the text in a column headed \"text\".");
  return { rows: out, notes };
}
