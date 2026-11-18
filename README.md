# Sliding-window intertextuality: experiment builder and molab notebook

Two parts that work together:

- **The website** (this folder, published with GitHub Pages) takes a query text and a corpus as CSV files and writes an **experiment file** (`.json`). It runs nothing itself, and the texts never leave the visitor's browser.
- **The notebook** (`notebook/intertextuality_molab.py`) runs on [molab](https://molab.marimo.io), marimo's free cloud notebooks, using the free GPU (an NVIDIA RTX Pro 6000 Blackwell with 96 GB). It reads the experiment file, slides a window across each corpus, and has **Qwen3.5-4B** and **Qwen3.5-27B** rate every window from 0 to 10. It produces the ranking of every window, a graph of relatedness along the corpus, the corpus text with a bar above every word, and the raw scores.

Without an experiment file, the notebook runs four built-in experiments. The query is Matthew 7:1-6, and the corpora are 1 Clement 13 (a true positive, because 13:2 quotes Matthew 7:1-2) and Psalm 25 (a true negative), each in Koine Greek and English.

The method follows Umphrey, Roberts & Roberts (2024), "Investigating Expert-in-the-Loop LLM Discourse Patterns for Ancient Intertextual Analysis" (NLP4DH 2024). The prompt and scoring are the same as in the earlier Colab notebooks.

## Publish it

1. Create a GitHub repository, for example `intertextuality`, and upload the **contents** of this folder to its root. Keep the `css`, `js`, `examples` and `notebook` folders and the `.nojekyll` file.
2. In the repository, open **Settings → Pages**. Choose **Deploy from a branch**, then the `main` branch and the `/ (root)` folder.
3. After a minute or two the site is live at `https://<your-username>.github.io/<repository>/`.

On a `github.io` address the site works out the repository from its own URL, so the **Open in molab** button goes straight to:

```
https://molab.marimo.io/github/<your-username>/<repository>/blob/main/notebook/intertextuality_molab.py
```

If the branch is not `main`, the site lives in a `docs/` folder, or the site is on another domain, set `GITHUB_REPO`, `GITHUB_BRANCH` and `NOTEBOOK_PATH` at the top of `js/app.js`.

For a badge in the repository's own README:

```markdown
[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/<your-username>/<repository>/blob/main/notebook/intertextuality_molab.py)
```

To try the site on your own computer, run `python -m http.server 8000` in this folder and open http://localhost:8000. Opening `index.html` straight from disk does not work, because the page loads the example files with `fetch`.

## Run an experiment

1. On the website, choose a query and a corpus (CSV files, or the examples). Set the window and step, optionally mark verses to shade on the graphs, and press **Add to the experiment file**. Repeat for as many pairs as you like. **Add the four built-in experiments** and **Add the 13 suggested experiments (Greek)** fill the list with ready-made ones.
2. Choose the models and press **Download the experiment file**.
3. Press **Open in molab** and sign in.
4. In molab, switch on the GPU with the **notebook specs** button in the header. The notebook restarts.
5. Run all the cells with the ▶ run button at the bottom right. The first cell installs a recent `transformers` if molab's is too old for Qwen3.5.
6. Drop the experiment file into the box in section 1. Uploaded files are kept between sessions only if they go through the **file tree** in molab's sidebar; a file uploaded there appears in the menu under the box after a restart.
7. Press **Run the experiments**. The first run downloads the models: about 9 GB for Qwen3.5-4B and 56 GB for Qwen3.5-27B. The scoring itself takes a few minutes; the four built-in experiments are 178 windows per model.
8. Download the results from section 6.

molab is free "as long as usage is reasonable". A notebook can run for up to 12 hours and shuts down after 90 minutes of inactivity. Scores are saved window by window in `outputs/window_scores.jsonl`. If a run is interrupted, pressing the button again carries on where it stopped, as long as the session is still alive.

## What the notebook produces

| File | Contents |
|---|---|
| `window_rankings.csv` | Every window with its rank, score, most likely answer, the probability of each answer 0–10 (`p_0` … `p_10`), the probability of answering with a number at all, its word range, references and text. |
| `word_scores.csv` | Every corpus word with each model's score: the mean of the windows that contain it. |
| `summary.csv` | One row per experiment and model: the highest window, the mean score, and mean scores inside and outside the marked verses. |
| `relatedness_<experiment>.png` / `.svg`, `relatedness_all.png` | Graphs of relatedness along the corpus, one line per model. |
| `reading_view.html` | The corpus text with a bar above every word for each model. It opens in any browser. |
| `raw_window_scores.jsonl`, `models.json`, `experiment_file.json` | The raw scores, how each model was loaded (including what it wrote when asked about the first window), and the experiments as run. |

## How a score is computed

1. The corpus is split into words. A window of the chosen length (default 40 words) starts at the first word and moves forward by the step (default 4). A final window is added if needed, so the last words are covered.
2. For each window, the model gets the whole query and the window's text, never the references, and is asked for one integer from 0 to 10. The prompt is in the notebook and on the website under "How the score is calculated".
3. Thinking is switched off in Qwen3.5's chat template (`enable_thinking=False`), so the reply starts with the answer.
4. One forward pass gives the probability of each possible first token. Qwen writes "10" as "1" followed by "0", so the notebook adds a "1" to the same pass and reads how likely "0" is next. The score is the probability-weighted mean of 0–10.
5. Each word's score on the graphs is the mean of the windows containing it.

The models run in bfloat16 on the GPU, one window at a time. Qwen3.5 mixes linear-attention (Gated DeltaNet) and full-attention layers, so nothing is cached between windows: every window is a complete forward pass. The notebook loads the official image-text-to-text class (`AutoModelForImageTextToText`) and stops if any weights are missing from the checkpoint.

**Checks.** The notebook was tested end to end in the marimo editor and headless, using tiny randomly initialised Qwen3.5 models with the real Qwen3.5 chat template. The test covered the upload, run, resume and download paths, and the scores matched an independent recomputation to five decimal places. It has not been run here with the real 4B and 27B weights, which could not be downloaded in this environment. The first molab run is that check: `models.json` records what each model actually wrote for the first window, and it should be a bare number.

## Other models

Any Hugging Face model whose tokenizer writes the digits 0–9 as single tokens can be added in the notebook or the website, for example the newer `Qwen/Qwen3.6-27B` or `Qwen/Qwen3.8-27B`, which use the same architecture. These have not been tested. Qwen3.5-27B needs about 60 GB of GPU memory in bfloat16.

## Experiment file format

```json
{
  "format": "sliding-window-intertextuality",
  "version": 1,
  "title": "…",
  "models": ["Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-27B"],
  "experiments": [
    {
      "id": "clem13-grc",
      "title": "1 Clement 13 · Greek",
      "kind": "true positive",
      "query":  {"name": "Matthew 7:1-6", "rows": [{"reference": "Matthew 7:1", "text": "Μὴ κρίνετε, …"}]},
      "corpus": {"name": "1 Clement 13", "rows": [{"reference": "1 Clement 13:1", "text": "…"}]},
      "window_words": 40,
      "stride_words": 4,
      "regions": [{"from": "1 Clement 13:2", "to": "1 Clement 13:2", "kind": "quotation", "label": "quotes Matt 7:1-2"}],
      "notes": "…"
    }
  ]
}
```

`kind`, `regions` and `notes` are optional. A region's `kind` is `quotation`, `allusion` or `overlap` (shared words, no dependence), and its `from` and `to` must match corpus references exactly. Windows with the same texts, settings and model are never scored twice, even across experiment files.

## CSV format

One row per verse, line or section, with a header row. The `text` column holds the words; an optional `reference` column labels each row.

```
reference,text
Matthew 7:1,"Judge not, that ye be not judged."
```

Save as UTF-8 ("CSV UTF-8" in spreadsheet programs) and put quotes around text that contains commas. Semicolon- and tab-separated files work too.

## Files

```
index.html, css/style.css       the page
js/app.js                       page logic: texts, experiment list, download, molab link
js/experiment.js                windows, reference parsing and the file format (no dependencies)
js/csv.js                       CSV reading
examples/*.csv                  Matthew 7:1-6 and 7:1-13, 1 Clement 13 and Psalm 25, in Greek and English
examples/default_experiments.json    the notebook's four built-in experiments
examples/suggested_experiments.json  13 further experiments in Greek (see below)
notebook/intertextuality_molab.py    the marimo notebook for molab
```

## Suggested experiments (`examples/suggested_experiments.json`)

All are in Greek, with the passages marked for the graphs. The query is Matthew 7:1-6 unless noted.

- **Known quotations and parallels (true positives):** Polycarp, *Philippians* 1–3 (2.3 quotes Matt 7:1-2); Luke 6:27-49 (the synoptic parallel); Didache 9–10 (9.5 quotes Matt 7:6); Psalm 6 LXX with query Matthew 7:21-23 (7:23 quotes Ps 6:9 LXX); Jeremiah 9:17-26 LXX and 1 Kingdoms 2:1-10 LXX, both with query 1 Clement 13:1 (Clement's composite "boast" quotation).
- **Probable or debated:** Mark 4:21-34 (4:24 has the measure saying in another context), Romans 2:1-16, James 4, Proverbs 11 LXX (11:22, the ring in the swine's snout).
- **Traps and true negatives:** 2 Peter 2:10-22 (dog and sow, from Proverbs 26:11), Psalm 26 (LXX 25, "Judge me, O Lord"), and 1 Clement 20 (same author as the true positive, no Jesus tradition).

## Sources and licences of the texts

- Greek New Testament: SBL Greek New Testament (SBLGNT), via [MorphGNT](https://github.com/morphgnt/sblgnt), under the [SBLGNT licence](https://sblgnt.com/license/).
- Apostolic Fathers in Greek (1 Clement, Polycarp, Didache): [Open Apostolic Fathers](https://github.com/jtauber/apostolic-fathers), edited by James Tauber and Seumas Macdonald after Lake (1912), CC BY-SA 4.0.
- 1 Clement 13, English: Kirsopp Lake, *The Apostolic Fathers*, vol. 1 (Loeb, 1912), public domain.
- Septuagint: Swete's edition (public domain), via [LXX-Swete-1930](https://github.com/eliranwong/LXX-Swete-1930). The Septuagint numbers Psalm 25 as 24 and Psalm 26 as 25; the files use English numbering, which divides these psalms the same way. Psalm 6 is labelled with English verse numbers (the Septuagint counts the title as verse 1).
- English Bible text: King James Version.
# intertextuality_project
