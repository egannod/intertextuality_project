# Intertextuality notebook builder

A static page that turns query and corpus CSV files into a sliding-window intertextuality notebook, ready to run in Google Colab (`.ipynb`) or molab (`.py`). Nothing is sent anywhere; the notebook is assembled in the browser.

## Publish on GitHub Pages

1. Upload the contents of this folder (`index.html`, `css`, `js`, `templates`, `examples`, `.nojekyll`) to the root of a GitHub repository.
2. **Settings → Pages →** Deploy from a branch, `main`, `/ (root)`.
3. The page is live at `https://<username>.github.io/<repository>/` after a minute or two.

To try it locally, run `python -m http.server 8000` here and open http://localhost:8000.

## Running the notebooks

- **Colab:** File → Upload notebook, choose a GPU runtime (Runtime → Change runtime type), then Runtime → Run all. The results download as a zip at the end.
- **molab:** create a new notebook, paste the whole `.py` file into its empty cell (the cells are created automatically), switch on the GPU, run all cells, then press **Run the experiments**.

Window size and step (40 and 4 words by default) are set near the top of each notebook.

## Files

```
index.html, css/style.css   the page
js/app.js                   uploads, experiments, models, downloads
js/notebooks.js             fills the templates with the texts, experiments and models
js/csv.js                   CSV reading
templates/colab.ipynb       the Colab notebook, with a "__DATA__" placeholder
templates/molab.py          the molab (marimo) notebook, with the same placeholder
examples/*.csv              example texts (SBLGNT, Swete's Septuagint, Lake's 1 Clement, KJV)
```

CSV format: a header row, a `text` column and optionally a `reference` column. Save as UTF-8.
