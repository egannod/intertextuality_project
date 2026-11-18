# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "accelerate",
#     "huggingface-hub",
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "torch",
#     "transformers>=5.23.0",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Intertextuality with Qwen3.5")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import datetime
    import gc
    import hashlib
    import html
    import importlib
    import importlib.metadata
    import io
    import json
    import math
    import os
    import re
    import shutil
    import subprocess
    import sys
    import time
    import unicodedata
    import zipfile
    from pathlib import Path
    return (
        Path, datetime, gc, hashlib, html, importlib, io, json, math, mo, os,
        re, shutil, subprocess, sys, time, unicodedata, zipfile,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Sliding-window intertextuality with Qwen3.5

    A window slides across a **corpus**, and a language model rates how closely each stretch relates to a **query** passage, from 0 (unrelated) to 10 (one quotes the other). The scores show where in the corpus the connection lies. The method follows Umphrey, Roberts & Roberts (2024), *Investigating Expert-in-the-Loop LLM Discourse Patterns for Ancient Intertextual Analysis* (NLP4DH 2024), with Qwen3.5-4B and Qwen3.5-27B in place of Claude.

    **How to run it on molab**

    1. **Turn on the GPU.** Click the notebook specs button in the header and choose the GPU. The notebook restarts with the GPU attached. Then **run all the cells** with the ▶ run button at the bottom right; the first cell installs anything missing.
    2. **Choose the experiments** in section 1. Drop in an experiment file made with the website, or keep the built-in experiments: Matthew 7:1-6 against 1 Clement 13 (a true positive, because 1 Clement 13:2 quotes Matthew 7:1-2) and against Psalm 25 (a true negative), each in Koine Greek and in English.
    3. **Press "Run the experiments"** in section 2. Qwen3.5-4B (9 GB) and Qwen3.5-27B (56 GB) download the first time they are used, which takes most of the time. The scoring itself takes a few minutes.
    4. **Read and download the results** in sections 3 to 5: the ranking of every window, a graph of relatedness along the corpus, the corpus text with a bar above every word, and a zip file with all of it.
    """)
    return


@app.cell(hide_code=True)
def _(Path, importlib, mo, re, shutil, subprocess, sys):
    # Install anything missing. Qwen3.5 needs a recent transformers (support arrived in 5.2; 5.23 is the tested version).
    MIN_TRANSFORMERS = "5.23.0"

    def _version(pkg):
        try:
            return importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            return None

    def _at_least(v, minimum):
        def _nums(s):
            return tuple(int(x) for x in re.findall(r"\d+", s)[:3])
        return v is not None and _nums(v) >= _nums(minimum)

    _needed = [] if _at_least(_version("transformers"), MIN_TRANSFORMERS) else [f"transformers>={MIN_TRANSFORMERS}"]
    _needed += [p for p in ("torch", "accelerate", "pandas", "numpy", "matplotlib") if _version(p) is None]
    install_note = ""
    if _needed:
        _cmds = [[sys.executable, "-m", "pip", "install", "--quiet", *_needed]]
        if shutil.which("uv"):
            _cmds.insert(0, ["uv", "pip", "install", "--python", sys.executable, *_needed])
        _errors = []
        for _cmd in _cmds:
            _r = subprocess.run(_cmd, capture_output=True, text=True)
            if _r.returncode == 0:
                install_note = "Installed " + ", ".join(_needed) + "."
                break
            _errors.append("$ " + " ".join(_cmd) + "\n" + _r.stderr[-2000:])
        else:
            raise RuntimeError("Could not install " + ", ".join(_needed) + ".\n\n" + "\n\n".join(_errors))
        importlib.invalidate_caches()

    _old = sys.modules.get("transformers")
    mo.stop(
        _old is not None and not _at_least(getattr(_old, "__version__", None), MIN_TRANSFORMERS),
        mo.callout(mo.md("A newer **transformers** was just installed, but the old one is already loaded. "
                         "Restart the notebook's kernel and run it again."), kind="danger"),
    )

    import numpy as np
    import pandas as pd
    import torch
    import transformers
    from huggingface_hub import constants as _hf_constants

    HAS_GPU = torch.cuda.is_available()
    DEVICE = "cuda" if HAS_GPU else "cpu"
    GPU_NAME = torch.cuda.get_device_name(0) if HAS_GPU else "none"
    VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9 if HAS_GPU else 0.0
    gpu_problem = ""
    if HAS_GPU:
        try:   # make sure this PyTorch build has kernels for this GPU
            _a = torch.ones(64, 64, device="cuda", dtype=torch.bfloat16)
            float((_a @ _a).sum())
        except Exception as _e:
            gpu_problem = f"PyTorch {torch.__version__} cannot run on this GPU ({type(_e).__name__}: {_e})."

    HF_CACHE = Path(_hf_constants.HF_HUB_CACHE)
    HF_CACHE.mkdir(parents=True, exist_ok=True)

    def free_disk_gb():
        return shutil.disk_usage(HF_CACHE).free / 1e9

    _rows = [
        ("GPU", f"{GPU_NAME} ({VRAM_GB:.0f} GB)" if HAS_GPU else "none"),
        ("Free disk for model downloads", f"{free_disk_gb():.0f} GB"),
        ("PyTorch", torch.__version__),
        ("transformers", transformers.__version__),
    ]
    _table = mo.md("| | |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in _rows))
    if gpu_problem:
        _status = mo.callout(mo.md(gpu_problem), kind="danger")
    elif not HAS_GPU:
        _status = mo.callout(mo.md(
            "**No GPU is attached.** Click the notebook specs button in the header and choose the GPU; the notebook "
            "restarts with it. Without a GPU only Qwen3.5-4B can run, very slowly, and Qwen3.5-27B is skipped."), kind="warn")
    else:
        _status = mo.callout(mo.md("The GPU is ready. " + install_note), kind="success")
    mo.vstack([mo.md("### Setup"), _status, _table])
    return DEVICE, HAS_GPU, HF_CACHE, VRAM_GB, free_disk_gb, gpu_problem, np, pd, torch, transformers


@app.cell(hide_code=True)
def _(json):
    # The prompt is identical to the Colab notebooks and the earlier website.
    PROMPT_TEMPLATE = """Compare the two passages below.

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

Answer with a single integer from 0 to 10 and nothing else."""

    SCORING_VERSION = "expected-score-v1"
    DEFAULT_MODELS = ["Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-27B"]
    MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#d6336c", "#b88400"]

    # The built-in experiments: Matthew 7:1-6 against 1 Clement 13 and Psalm 25, in Greek and English.
    DEFAULT_FILE = json.loads(r"""{
 "format": "sliding-window-intertextuality",
 "version": 1,
 "title": "Matthew 7:1-6 against 1 Clement 13 and Psalm 25",
 "created_with": "built into the notebook",
 "models": [
  "Qwen/Qwen3.5-4B",
  "Qwen/Qwen3.5-27B"
 ],
 "experiments": [
  {
   "id": "clem13-grc",
   "title": "1 Clement 13 · Greek",
   "kind": "true positive",
   "query": {
    "name": "Matthew 7:1-6",
    "rows": [
     {
      "reference": "Matthew 7:1",
      "text": "Μὴ κρίνετε, ἵνα μὴ κριθῆτε·"
     },
     {
      "reference": "Matthew 7:2",
      "text": "ἐν ᾧ γὰρ κρίματι κρίνετε κριθήσεσθε, καὶ ἐν ᾧ μέτρῳ μετρεῖτε μετρηθήσεται ὑμῖν."
     },
     {
      "reference": "Matthew 7:3",
      "text": "τί δὲ βλέπεις τὸ κάρφος τὸ ἐν τῷ ὀφθαλμῷ τοῦ ἀδελφοῦ σου, τὴν δὲ ἐν τῷ σῷ ὀφθαλμῷ δοκὸν οὐ κατανοεῖς;"
     },
     {
      "reference": "Matthew 7:4",
      "text": "ἢ πῶς ἐρεῖς τῷ ἀδελφῷ σου· Ἄφες ἐκβάλω τὸ κάρφος ἐκ τοῦ ὀφθαλμοῦ σου, καὶ ἰδοὺ ἡ δοκὸς ἐν τῷ ὀφθαλμῷ σοῦ;"
     },
     {
      "reference": "Matthew 7:5",
      "text": "ὑποκριτά, ἔκβαλε πρῶτον ἐκ τοῦ ὀφθαλμοῦ σοῦ τὴν δοκόν, καὶ τότε διαβλέψεις ἐκβαλεῖν τὸ κάρφος ἐκ τοῦ ὀφθαλμοῦ τοῦ ἀδελφοῦ σου."
     },
     {
      "reference": "Matthew 7:6",
      "text": "Μὴ δῶτε τὸ ἅγιον τοῖς κυσίν, μηδὲ βάλητε τοὺς μαργαρίτας ὑμῶν ἔμπροσθεν τῶν χοίρων, μήποτε καταπατήσουσιν αὐτοὺς ἐν τοῖς ποσὶν αὐτῶν καὶ στραφέντες ῥήξωσιν ὑμᾶς."
     }
    ]
   },
   "corpus": {
    "name": "1 Clement 13",
    "rows": [
     {
      "reference": "1 Clement 13:1",
      "text": "Ταπεινοφρονήσωμεν οὖν, ἀδελφοί, ἀποθέμενοι πᾶσαν ἀλαζονείαν καὶ τῦφος καὶ ἀφροσύνην καὶ ὀργάς, καὶ ποιήσωμεν τὸ γεγραμμένον, λέγει γὰρ τὸ πνεῦμα τὸ ἅγιον· Μὴ καυχάσθω ὁ σοφὸς ἐν τῇ σοφίᾳ αὐτοῦ μηδὲ ὁ ἰσχυρὸς ἐν τῇ ἰσχύϊ αὐτοῦ μηδὲ ὁ πλούσιος ἐν τῷ πλούτῳ αὐτοῦ, ἀλλ’ ἤ ὁ καυχώμενος ἐν κυρίῳ καυχάσθω, τοῦ ἐκζητεῖν αὐτὸν καὶ ποιεῖν κρίμα καὶ δικαιοσύνην· μάλιστα μεμνημένοι τῶν λόγων τοῦ κυρίου Ἰησοῦ, οὓς ἐλάλησεν διδάσκων ἐπιείκειαν καὶ μακροθυμίαν."
     },
     {
      "reference": "1 Clement 13:2",
      "text": "οὕτως γὰρ εἶπεν· Ἐλεᾶτε, ἵνα ἐλεηθῆτε· ἀφίετε, ἵνα ἀφεθῇ ὑμῖν· ὡς ποιεῖτε, οὕτω ποιηθήσεται ὑμῖν· ὡς δίδοτε, οὕτως δοθήσεται ὑμῖν· ὡς κρίνετε, οὕτως κριθήσεσθε· ὡς χρηστεύεσθε, οὕτως χρηστευθήσεται ὑμῖν· ᾧ μέτρῳ μετρεῖτε, ἐν αὐτῷ μετρηθήσεται ὑμῖν."
     },
     {
      "reference": "1 Clement 13:3",
      "text": "ταύτῃ τῇ ἐντολῇ καὶ τοῖς παραγγέλμασιν τούτοις στηρίξωμεν ἑαυτοὺς εἰς τὸ πορεύεσθαι ὑπηκόους ὄντας τοῖς ἁγιοπρεπέσι λόγοις αὐτοῦ, ταπεινοφρονοῦντες· φησὶν γὰρ ὁ ἅγιος λόγος·"
     },
     {
      "reference": "1 Clement 13:4",
      "text": "Ἐπὶ τίνα ἐπιβλέψω, ἀλλ’ ἢ ἐπὶ τὸν πραῢν καὶ ἡσύχιον καὶ τρέμοντά μου τὰ λόγια."
     }
    ]
   },
   "window_words": 40,
   "stride_words": 4,
   "regions": [
    {
     "from": "1 Clement 13:2",
     "to": "1 Clement 13:2",
     "kind": "quotation",
     "label": "quotes Matt 7:1-2"
    },
    {
     "from": "1 Clement 13:1",
     "to": "1 Clement 13:1",
     "kind": "overlap",
     "label": "τὸ ἅγιον, κρίμα"
    }
   ],
   "notes": "1 Clement 13:2 quotes a string of sayings of Jesus, among them Matthew 7:1-2 (ὡς κρίνετε, οὕτως κριθήσεσθε … ᾧ μέτρῳ μετρεῖτε, ἐν αὐτῷ μετρηθήσεται ὑμῖν). 13:1 quotes Jeremiah 9:23-24 and 13:4 Isaiah 66:2; 13:1 shares τὸ ἅγιον and κρίμα with the query only by chance. Greek: SBLGNT (MorphGNT); 1 Clement, Open Apostolic Fathers (Lake 1912 text, CC BY-SA 4.0); Psalm, Swete's LXX."
  },
  {
   "id": "ps25-grc",
   "title": "Psalm 25 · Greek",
   "kind": "true negative",
   "query": {
    "name": "Matthew 7:1-6",
    "rows": [
     {
      "reference": "Matthew 7:1",
      "text": "Μὴ κρίνετε, ἵνα μὴ κριθῆτε·"
     },
     {
      "reference": "Matthew 7:2",
      "text": "ἐν ᾧ γὰρ κρίματι κρίνετε κριθήσεσθε, καὶ ἐν ᾧ μέτρῳ μετρεῖτε μετρηθήσεται ὑμῖν."
     },
     {
      "reference": "Matthew 7:3",
      "text": "τί δὲ βλέπεις τὸ κάρφος τὸ ἐν τῷ ὀφθαλμῷ τοῦ ἀδελφοῦ σου, τὴν δὲ ἐν τῷ σῷ ὀφθαλμῷ δοκὸν οὐ κατανοεῖς;"
     },
     {
      "reference": "Matthew 7:4",
      "text": "ἢ πῶς ἐρεῖς τῷ ἀδελφῷ σου· Ἄφες ἐκβάλω τὸ κάρφος ἐκ τοῦ ὀφθαλμοῦ σου, καὶ ἰδοὺ ἡ δοκὸς ἐν τῷ ὀφθαλμῷ σοῦ;"
     },
     {
      "reference": "Matthew 7:5",
      "text": "ὑποκριτά, ἔκβαλε πρῶτον ἐκ τοῦ ὀφθαλμοῦ σοῦ τὴν δοκόν, καὶ τότε διαβλέψεις ἐκβαλεῖν τὸ κάρφος ἐκ τοῦ ὀφθαλμοῦ τοῦ ἀδελφοῦ σου."
     },
     {
      "reference": "Matthew 7:6",
      "text": "Μὴ δῶτε τὸ ἅγιον τοῖς κυσίν, μηδὲ βάλητε τοὺς μαργαρίτας ὑμῶν ἔμπροσθεν τῶν χοίρων, μήποτε καταπατήσουσιν αὐτοὺς ἐν τοῖς ποσὶν αὐτῶν καὶ στραφέντες ῥήξωσιν ὑμᾶς."
     }
    ]
   },
   "corpus": {
    "name": "Psalm 25",
    "rows": [
     {
      "reference": "Psalm 25:1",
      "text": "Πρὸς σέ, Κύριε, ἦρα τὴν ψυχήν μου,"
     },
     {
      "reference": "Psalm 25:2",
      "text": "ὁ θεός μου· ἐπὶ σοὶ πέποιθα· μὴ καταισχυνθείην, μηδὲ καταγελασάτωσάν μου οἱ ἐχθροί μου."
     },
     {
      "reference": "Psalm 25:3",
      "text": "καὶ γὰρ πάντες οἱ ὑπομένοντές σε οὐ μὴ καταισχυνθῶσιν· αἰσχυνθήτωσαν οἱ ἀνομοῦντες διὰ κενῆς."
     },
     {
      "reference": "Psalm 25:4",
      "text": "τὰς ὁδούς σου, Κύριε, γνώρισόν μοι, καὶ τὰς τρίβους σου δίδαξόν με·"
     },
     {
      "reference": "Psalm 25:5",
      "text": "ὁδήγησόν με ἐπὶ τὴν ἀλήθειάν σου καὶ δίδαξόν με, ὅτι σὺ εἶ ὁ θεὸς ὁ σωτήρ μου, καὶ σὲ ὑπέμεινα ὅλην τὴν ἡμέραν."
     },
     {
      "reference": "Psalm 25:6",
      "text": "μνήσθητι τῶν οἰκτειρμῶν σου, καὶ τὰ ἐλέη σου, ὅτι ἀπὸ τοῦ αἰῶνός εἰσιν."
     },
     {
      "reference": "Psalm 25:7",
      "text": "ἁμαρτίας νεότητός μου καὶ ἀγνοίας μὴ μνησθῇς· κατὰ τὸ ἔλεός σου μνήσθητί μου ἕνεκα τῆς χρηστότητός σου, Κύριε."
     },
     {
      "reference": "Psalm 25:8",
      "text": "χρηστὸς καὶ εὐθὴς ὁ κύριος· διὰ τοῦτο νομοθετήσει ἁμαρτάνοντας ἐν ὁδῷ."
     },
     {
      "reference": "Psalm 25:9",
      "text": "ὁδηγήσει πρᾳεῖς ἐν κρίσει, διδάξει πρᾳεῖς ὁδοὺς αὐτοῦ."
     },
     {
      "reference": "Psalm 25:10",
      "text": "πᾶσαι αἱ ὁδοὶ Κυρίου ἔλεος καὶ ἀλήθεια τοῖς ἐκζητοῦσιν τὴν διαθήκην αὐτοῦ καὶ τὰ μαρτύρια αὐτοῦ."
     },
     {
      "reference": "Psalm 25:11",
      "text": "ἕνεκα τοῦ ὀνόματός σου, Κύριε, καὶ ἱλάσῃ τῇ ἁμαρτίᾳ μου· πολλὴ γάρ ἐστιν."
     },
     {
      "reference": "Psalm 25:12",
      "text": "τίς ἐστιν ἄνθρωπος ὁ φοβούμενος τὸν κύριον; νομοθετήσει αὐτῷ ἐν ὁδῷ ᾗ ᾑρετίσατο·"
     },
     {
      "reference": "Psalm 25:13",
      "text": "ἡ ψυχὴ αὐτοῦ ἐν ἀγαθοῖς αὐλισθήσεται, καὶ τὸ σπέρμα αὐτοῦ κληρονομήσει γῆν."
     },
     {
      "reference": "Psalm 25:14",
      "text": "κραταίωμα Κύριος τῶν φοβουμένων αὐτόν, καὶ τὸ ὄνομα Κυρίου τῶν φοβουμένων αὐτόν, καὶ ἡ διαθήκη αὐτοῦ τοῦ δηλῶσαι αὐτοῖς·"
     },
     {
      "reference": "Psalm 25:15",
      "text": "οἱ ὀφθαλμοί μου διὰ παντὸς πρὸς τὸν κύριον, ὅτι αὐτὸς ἐκσπάσει ἐκ παγίδος τοὺς πόδας μου."
     },
     {
      "reference": "Psalm 25:16",
      "text": "ἐπίβλεψον ἐπ᾽ ἐμὲ καὶ ἐλέησόν με, ὅτι μονογενὴς καὶ πτωχός εἰμι ἐγώ."
     },
     {
      "reference": "Psalm 25:17",
      "text": "αἱ θλίψεις τῆς καρδίας μου ἐπληθύνθησαν, ἐκ τῶν ἀναγκῶν μου ἐξάγαγέ με."
     },
     {
      "reference": "Psalm 25:18",
      "text": "ἴδε τὴν ταπείνωσίν μου καὶ τὸν κόπον μου, καὶ ἄφες πάσας τὰς ἁμαρτίας μου."
     },
     {
      "reference": "Psalm 25:19",
      "text": "ἴδε τοὺς ἐχθρούς μου, ὅτι ἐπληθύνθησαν, καὶ μῖσος ἄδικον ἐμίσησάν με."
     },
     {
      "reference": "Psalm 25:20",
      "text": "φύλαξον τὴν ψυχήν μου καὶ ῥῦσαί με· μὴ καταισχυνθείην, ὅτι ἤλπισα ἐπὶ σέ."
     },
     {
      "reference": "Psalm 25:21",
      "text": "ἄκακοι καὶ εὐθεῖς ἐκολλῶντό μοι, ὅτι ὑπέμεινά σε, Κύριε."
     },
     {
      "reference": "Psalm 25:22",
      "text": "λύτρωσαι, ὁ θεός, τὸν Ἰσραὴλ ἐκ πασῶν τῶν θλίψεων αὐτοῦ."
     }
    ]
   },
   "window_words": 40,
   "stride_words": 4,
   "regions": [
    {
     "from": "Psalm 25:9",
     "to": "Psalm 25:9",
     "kind": "overlap",
     "label": "κρίσει"
    },
    {
     "from": "Psalm 25:15",
     "to": "Psalm 25:16",
     "kind": "overlap",
     "label": "ὀφθαλμοί, πόδας, ἐπίβλεψον"
    },
    {
     "from": "Psalm 25:18",
     "to": "Psalm 25:18",
     "kind": "overlap",
     "label": "ἄφες"
    }
   ],
   "notes": "Psalm 25 has no literary relationship with Matthew 7:1-6. It shares a few words: judgment (25:9), eyes and feet (25:15), looking (ἐπίβλεψον 25:16; English “turn”, cf. 7:6) and ἄφες (25:18, “forgive”, the same form as “let me” in 7:4). The Greek is Septuagint Psalm 24, which English Bibles number 25. Greek: SBLGNT (MorphGNT); 1 Clement, Open Apostolic Fathers (Lake 1912 text, CC BY-SA 4.0); Psalm, Swete's LXX."
  },
  {
   "id": "clem13-eng",
   "title": "1 Clement 13 · English",
   "kind": "true positive",
   "query": {
    "name": "Matthew 7:1-6",
    "rows": [
     {
      "reference": "Matthew 7:1",
      "text": "Judge not, that ye be not judged."
     },
     {
      "reference": "Matthew 7:2",
      "text": "For with what judgment ye judge, ye shall be judged: and with what measure ye mete, it shall be measured to you again."
     },
     {
      "reference": "Matthew 7:3",
      "text": "And why beholdest thou the mote that is in thy brother’s eye, but considerest not the beam that is in thine own eye?"
     },
     {
      "reference": "Matthew 7:4",
      "text": "Or how wilt thou say to thy brother, Let me pull out the mote out of thine eye; and, behold, a beam is in thine own eye?"
     },
     {
      "reference": "Matthew 7:5",
      "text": "Thou hypocrite, first cast out the beam out of thine own eye; and then shalt thou see clearly to cast out the mote out of thy brother’s eye."
     },
     {
      "reference": "Matthew 7:6",
      "text": "Give not that which is holy unto the dogs, neither cast ye your pearls before swine, lest they trample them under their feet, and turn again and rend you."
     }
    ]
   },
   "corpus": {
    "name": "1 Clement 13",
    "rows": [
     {
      "reference": "1 Clement 13:1",
      "text": "Let us, therefore, be humble-minded, brethren, putting aside all arrogance and conceit and foolishness and wrath, and let us do that which is written (for the Holy Spirit says, “Let not the wise man boast himself in his wisdom, nor the strong man in his strength, nor the rich man in his riches, but he that boasteth let him boast in the Lord, to seek him out and to do judgment and righteousness”), especially remembering the words of the Lord Jesus which he spoke when he was teaching gentleness and longsuffering."
     },
     {
      "reference": "1 Clement 13:2",
      "text": "For he spoke thus: “Be merciful, that ye may obtain mercy. Forgive, that ye may be forgiven. As ye do, so shall it be done unto you. As ye give, so shall it be given unto you. As ye judge, so shall ye be judged. As ye are kind, so shall kindness be shewn you. With what measure ye mete, it shall be measured to you.”"
     },
     {
      "reference": "1 Clement 13:3",
      "text": "With this commandment and with these injunctions let us strengthen ourselves to walk in obedience to his hallowed words and let us be humble-minded, for the holy word says,"
     },
     {
      "reference": "1 Clement 13:4",
      "text": "“On whom shall I look, but on the meek and gentle and him who trembles at my oracles.”"
     }
    ]
   },
   "window_words": 56,
   "stride_words": 6,
   "regions": [
    {
     "from": "1 Clement 13:2",
     "to": "1 Clement 13:2",
     "kind": "quotation",
     "label": "quotes Matt 7:1-2"
    },
    {
     "from": "1 Clement 13:1",
     "to": "1 Clement 13:1",
     "kind": "overlap",
     "label": "“holy”, “judgment”"
    }
   ],
   "notes": "1 Clement 13:2 quotes a string of sayings of Jesus, among them Matthew 7:1-2 (ὡς κρίνετε, οὕτως κριθήσεσθε … ᾧ μέτρῳ μετρεῖτε, ἐν αὐτῷ μετρηθήσεται ὑμῖν). 13:1 quotes Jeremiah 9:23-24 and 13:4 Isaiah 66:2; 13:1 shares τὸ ἅγιον and κρίμα with the query only by chance. English needs 1.39 times as many words as Greek for this text, so the window (56 words) and step (6) are scaled to cover about the same stretch as the Greek. English: King James Version; 1 Clement, Kirsopp Lake (1912)."
  },
  {
   "id": "ps25-eng",
   "title": "Psalm 25 · English",
   "kind": "true negative",
   "query": {
    "name": "Matthew 7:1-6",
    "rows": [
     {
      "reference": "Matthew 7:1",
      "text": "Judge not, that ye be not judged."
     },
     {
      "reference": "Matthew 7:2",
      "text": "For with what judgment ye judge, ye shall be judged: and with what measure ye mete, it shall be measured to you again."
     },
     {
      "reference": "Matthew 7:3",
      "text": "And why beholdest thou the mote that is in thy brother’s eye, but considerest not the beam that is in thine own eye?"
     },
     {
      "reference": "Matthew 7:4",
      "text": "Or how wilt thou say to thy brother, Let me pull out the mote out of thine eye; and, behold, a beam is in thine own eye?"
     },
     {
      "reference": "Matthew 7:5",
      "text": "Thou hypocrite, first cast out the beam out of thine own eye; and then shalt thou see clearly to cast out the mote out of thy brother’s eye."
     },
     {
      "reference": "Matthew 7:6",
      "text": "Give not that which is holy unto the dogs, neither cast ye your pearls before swine, lest they trample them under their feet, and turn again and rend you."
     }
    ]
   },
   "corpus": {
    "name": "Psalm 25",
    "rows": [
     {
      "reference": "Psalm 25:1",
      "text": "Unto thee, O LORD, do I lift up my soul."
     },
     {
      "reference": "Psalm 25:2",
      "text": "O my God, I trust in thee: let me not be ashamed, let not mine enemies triumph over me."
     },
     {
      "reference": "Psalm 25:3",
      "text": "Yea, let none that wait on thee be ashamed: let them be ashamed which transgress without cause."
     },
     {
      "reference": "Psalm 25:4",
      "text": "Shew me thy ways, O LORD; teach me thy paths."
     },
     {
      "reference": "Psalm 25:5",
      "text": "Lead me in thy truth, and teach me: for thou art the God of my salvation; on thee do I wait all the day."
     },
     {
      "reference": "Psalm 25:6",
      "text": "Remember, O LORD, thy tender mercies and thy lovingkindnesses; for they have been ever of old."
     },
     {
      "reference": "Psalm 25:7",
      "text": "Remember not the sins of my youth, nor my transgressions: according to thy mercy remember thou me for thy goodness’ sake, O LORD."
     },
     {
      "reference": "Psalm 25:8",
      "text": "Good and upright is the LORD: therefore will he teach sinners in the way."
     },
     {
      "reference": "Psalm 25:9",
      "text": "The meek will he guide in judgment: and the meek will he teach his way."
     },
     {
      "reference": "Psalm 25:10",
      "text": "All the paths of the LORD are mercy and truth unto such as keep his covenant and his testimonies."
     },
     {
      "reference": "Psalm 25:11",
      "text": "For thy name’s sake, O LORD, pardon mine iniquity; for it is great."
     },
     {
      "reference": "Psalm 25:12",
      "text": "What man is he that feareth the LORD? him shall he teach in the way that he shall choose."
     },
     {
      "reference": "Psalm 25:13",
      "text": "His soul shall dwell at ease; and his seed shall inherit the earth."
     },
     {
      "reference": "Psalm 25:14",
      "text": "The secret of the LORD is with them that fear him; and he will shew them his covenant."
     },
     {
      "reference": "Psalm 25:15",
      "text": "Mine eyes are ever toward the LORD; for he shall pluck my feet out of the net."
     },
     {
      "reference": "Psalm 25:16",
      "text": "Turn thee unto me, and have mercy upon me; for I am desolate and afflicted."
     },
     {
      "reference": "Psalm 25:17",
      "text": "The troubles of my heart are enlarged: O bring thou me out of my distresses."
     },
     {
      "reference": "Psalm 25:18",
      "text": "Look upon mine affliction and my pain; and forgive all my sins."
     },
     {
      "reference": "Psalm 25:19",
      "text": "Consider mine enemies; for they are many; and they hate me with cruel hatred."
     },
     {
      "reference": "Psalm 25:20",
      "text": "O keep my soul, and deliver me: let me not be ashamed; for I put my trust in thee."
     },
     {
      "reference": "Psalm 25:21",
      "text": "Let integrity and uprightness preserve me; for I wait on thee."
     },
     {
      "reference": "Psalm 25:22",
      "text": "Redeem Israel, O God, out of all his troubles."
     }
    ]
   },
   "window_words": 47,
   "stride_words": 5,
   "regions": [
    {
     "from": "Psalm 25:9",
     "to": "Psalm 25:9",
     "kind": "overlap",
     "label": "“judgment”"
    },
    {
     "from": "Psalm 25:15",
     "to": "Psalm 25:16",
     "kind": "overlap",
     "label": "“eyes”, “feet”, “turn”"
    }
   ],
   "notes": "Psalm 25 has no literary relationship with Matthew 7:1-6. It shares a few words: judgment (25:9), eyes and feet (25:15), looking (ἐπίβλεψον 25:16; English “turn”, cf. 7:6) and ἄφες (25:18, “forgive”, the same form as “let me” in 7:4). The Greek is Septuagint Psalm 24, which English Bibles number 25. English needs 1.18 times as many words as Greek for this text, so the window (47 words) and step (5) are scaled to cover about the same stretch as the Greek. English: King James Version; 1 Clement, Kirsopp Lake (1912)."
  }
 ]
}""")
    return DEFAULT_FILE, DEFAULT_MODELS, MODEL_COLORS, PROMPT_TEMPLATE, SCORING_VERSION


@app.cell(hide_code=True)
def _(DEFAULT_MODELS, PROMPT_TEMPLATE, SCORING_VERSION, hashlib, json, os, re, unicodedata):
    def nfc(s):
        return unicodedata.normalize("NFC", str(s))

    def build_prompt(query_text, excerpt):
        return PROMPT_TEMPLATE.format(query=query_text, excerpt=excerpt)

    def make_windows(n, size, stride):
        """Start positions of the windows; a final window is added if needed so the last words are covered."""
        size = max(1, min(size, n))
        starts = list(range(0, n - size + 1, max(1, stride)))
        if starts[-1] + size < n:
            starts.append(n - size)
        return [(s, s + size) for s in starts]

    def common_prefix(refs):
        """The shared beginning of the references, cut back to a space or colon ("Psalm 25:" for Psalm 25:1-22)."""
        if len(refs) < 2:
            return ""
        p = os.path.commonprefix(list(refs))
        cut = max(p.rfind(":"), p.rfind(" "), p.rfind("."))
        return p[:cut + 1] if cut >= 0 else ""

    def short_ref(ref, prefix):
        return ref[len(prefix):] if prefix and ref.startswith(prefix) and len(ref) > len(prefix) else ref

    def ref_range(a, b):
        if a == b:
            return a
        p = common_prefix([a, b])
        return f"{a}–{b[len(p):]}" if p.endswith(":") else f"{a} – {b}"

    def model_label(model_id):
        return model_id.rstrip("/").split("/")[-1]

    _REGION_KINDS = {"quotation": "quotation", "primary": "quotation", "allusion": "allusion", "echo": "allusion",
                     "overlap": "overlap", "superficial": "overlap", "secondary": "overlap"}

    def _passage(p, where):
        """{"name", "rows": [{"reference", "text"}]}, {"name", "text"} or a plain string -> rows with NFC text."""
        if isinstance(p, str):
            p = {"text": p}
        if not isinstance(p, dict):
            raise ValueError(f"{where}: missing")
        rows = p.get("rows")
        if rows is None and isinstance(p.get("text"), str):
            rows = [{"reference": "", "text": p["text"]}]
        if not isinstance(rows, list):
            raise ValueError(f"{where}: no rows of text")
        out = []
        for i, r in enumerate(rows, 1):
            if isinstance(r, str):
                r = {"text": r}
            text = " ".join(nfc(r.get("text", "")).split()) if isinstance(r, dict) else ""
            if text:
                ref = nfc(r.get("reference") or "").strip() or f"row {i}"
                out.append({"reference": ref, "text": text})
        if not out:
            raise ValueError(f"{where}: the text is empty")
        name = nfc(p.get("name") or "").strip() or out[0]["reference"]
        return {"name": name, "rows": out}

    def read_experiment_file(obj, origin):
        """Check an experiment file and work out every window. Raises ValueError with a readable message."""
        if not isinstance(obj, dict) or not isinstance(obj.get("experiments"), list) or not obj["experiments"]:
            raise ValueError("This file has no list of experiments. Make it with the website, or check that it is the right file.")
        notes = []
        models = [m.strip() for m in (obj.get("models") or []) if isinstance(m, str) and m.strip()]
        models = list(dict.fromkeys(models)) or list(DEFAULT_MODELS)
        experiments, ids, titles = [], set(), set()
        for i, e in enumerate(obj["experiments"], 1):
            if not isinstance(e, dict):
                raise ValueError(f"Experiment {i} is not an object.")
            query = _passage(e.get("query"), f"Experiment {i}, query")
            corpus = _passage(e.get("corpus"), f"Experiment {i}, corpus")
            words, word_row = [], []
            for k, r in enumerate(corpus["rows"]):
                for w in r["text"].split():
                    words.append(w)
                    word_row.append(k)
            try:
                size = int(e.get("window_words", 40))
                stride = int(e.get("stride_words", 4))
            except (TypeError, ValueError):
                raise ValueError(f"Experiment {i}: window_words and stride_words must be whole numbers.")
            if size < 2 or stride < 1:
                raise ValueError(f"Experiment {i}: the window must be at least 2 words and the step at least 1.")
            if size > len(words):
                notes.append(f"Experiment {i}: the window ({size} words) is longer than the corpus ({len(words)} words), so it covers the whole corpus.")
                size = len(words)
            eid = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(e.get("id") or f"exp{i}")).strip("-") or f"exp{i}"
            while eid in ids:
                eid += "b"
            ids.add(eid)
            title = nfc(e.get("title") or "").strip() or f"{query['name']} → {corpus['name']}"
            while title in titles:
                title += f" ({eid})"
            titles.add(title)
            refs = [r["reference"] for r in corpus["rows"]]
            regions = []
            for g in e.get("regions") or []:
                try:
                    a, b = refs.index(nfc(g["from"]).strip()), refs.index(nfc(g.get("to") or g["from"]).strip())
                    kind = _REGION_KINDS[str(g.get("kind") or g.get("role") or "quotation").lower()]
                except (KeyError, ValueError, TypeError, AttributeError):
                    notes.append(f"Experiment {i}: a marked region does not match the corpus references and was left out.")
                    continue
                regions.append({"rows": (min(a, b), max(a, b) + 1), "kind": kind, "label": nfc(g.get("label") or "")})
            query_text = " ".join(r["text"] for r in query["rows"])
            windows = []
            for n, (s, t) in enumerate(make_windows(len(words), size, stride), 1):
                windows.append({"window": n, "start": s, "end": t,
                                "references": ref_range(refs[word_row[s]], refs[word_row[t - 1]]),
                                "text": " ".join(words[s:t])})
            fingerprint = hashlib.sha1(json.dumps([SCORING_VERSION, PROMPT_TEMPLATE, query_text, words, size, stride],
                                                  ensure_ascii=False).encode()).hexdigest()[:12]
            experiments.append({
                "id": eid, "title": title, "kind": nfc(e.get("kind") or "").strip(), "notes": nfc(e.get("notes") or ""),
                "query": query, "query_text": query_text, "corpus": corpus, "refs": refs,
                "words": words, "word_row": word_row, "window_words": size, "stride_words": stride,
                "windows": windows, "regions": regions, "fingerprint": fingerprint,
            })
        return {"origin": origin, "title": nfc(obj.get("title") or ""), "models": models,
                "experiments": experiments, "notes": notes, "raw": obj}

    def looks_like_experiment_file(path):
        try:
            if path.stat().st_size > 20_000_000:
                return False
            obj = json.loads(path.read_text(encoding="utf-8"))
            return isinstance(obj, dict) and isinstance(obj.get("experiments"), list)
        except Exception:
            return False

    return (build_prompt, common_prefix, looks_like_experiment_file, model_label, nfc, read_experiment_file,
            ref_range, short_ref)


@app.cell(hide_code=True)
def _(Path, looks_like_experiment_file, mo):
    NB_DIR = Path(mo.notebook_dir() or ".").resolve()
    OUT_DIR = NB_DIR / "outputs"

    _found = []
    for _folder in [NB_DIR, *[p for p in NB_DIR.iterdir() if p.is_dir() and not p.name.startswith((".", "_")) and p.name != "outputs"]]:
        try:
            _found += [p for p in sorted(_folder.glob("*.json")) if looks_like_experiment_file(p)]
        except OSError:
            pass

    BUILTIN_LABEL = "Built-in: Matthew 7:1-6 against 1 Clement 13 and Psalm 25 (Greek and English)"
    upload = mo.ui.file(filetypes=[".json"], kind="area",
                        label="Drop the experiment file from the website here, or click to choose it")
    source = mo.ui.dropdown(
        options={BUILTIN_LABEL: "builtin", **{f"File in this notebook's folder: {p.relative_to(NB_DIR)}": str(p) for p in _found}},
        value=BUILTIN_LABEL, label="Or use")
    mo.vstack([
        mo.md("## 1. Choose the experiments"),
        mo.md("An experiment file from the website holds the query, the corpus and the window settings. "
              "A file you dropped here wins over the menu below it. Files uploaded through the file tree in the "
              "sidebar are kept between sessions; they appear in the menu after the notebook restarts."),
        upload, source,
    ])
    return NB_DIR, OUT_DIR, source, upload


@app.cell(hide_code=True)
def _(DEFAULT_FILE, Path, json, mo, pd, read_experiment_file, source, upload):
    try:
        if upload.value:
            _obj, _origin = json.loads(upload.contents(0).decode("utf-8-sig")), f"uploaded file {upload.name(0)}"
        elif source.value != "builtin":
            _obj, _origin = json.loads(Path(source.value).read_text(encoding="utf-8-sig")), Path(source.value).name
        else:
            _obj, _origin = DEFAULT_FILE, "built-in experiments"
        plan = read_experiment_file(_obj, _origin)
    except (ValueError, UnicodeDecodeError) as _e:
        mo.stop(True, mo.callout(mo.md(f"**This experiment file cannot be used.** {_e}"), kind="danger"))

    _overview = pd.DataFrame([{
        "experiment": e["title"],
        "query": f'{e["query"]["name"]} ({len(e["query_text"].split())} words)',
        "corpus": f'{e["corpus"]["name"]} ({len(e["words"])} words)',
        "window": f'{e["window_words"]} words',
        "step": f'{e["stride_words"]} words',
        "windows": len(e["windows"]),
    } for e in plan["experiments"]])
    mo.vstack([
        mo.md(f"**Using {plan['origin']}**" + (f": {plan['title']}" if plan["title"] else "")
              + f" · {len(plan['experiments'])} experiments · models in the file: "
              + ", ".join(f"`{m}`" for m in plan["models"])),
        *[mo.callout(mo.md(n), kind="warn") for n in plan["notes"]],
        mo.ui.table(_overview, selection=None, show_download=False, show_column_summaries=False),
    ])
    return (plan,)


@app.cell(hide_code=True)
def _(DEFAULT_MODELS, mo, plan):
    _options = list(dict.fromkeys(DEFAULT_MODELS + plan["models"]))
    models_ui = mo.ui.multiselect(options=_options, value=[m for m in plan["models"] if m in _options],
                                  label="Models", full_width=True)
    extra_model_ui = mo.ui.text(placeholder="for example Qwen/Qwen3.6-27B", full_width=True,
                                label="Another Hugging Face model (optional)")
    experiments_ui = mo.ui.multiselect(options={e["title"]: e["id"] for e in plan["experiments"]},
                                       value=[e["title"] for e in plan["experiments"]], label="Experiments",
                                       full_width=True)
    token_ui = mo.ui.text(kind="password", label="Hugging Face token (optional: faster, more reliable downloads)",
                          full_width=True)
    cleanup_ui = mo.ui.checkbox(value=False, label="Delete each model's downloaded files when it is finished "
                                                   "(frees disk space; it downloads again next time)")
    run_button = mo.ui.run_button(label="Run the experiments", kind="success")
    mo.vstack([
        mo.md("## 2. Run"),
        models_ui, extra_model_ui, experiments_ui, token_ui, cleanup_ui, run_button,
        mo.md("Scores are saved window by window in the `outputs` folder. If the run stops, press the button again "
              "and it carries on where it left off; windows already scored with the same texts and settings are "
              "never scored twice."),
    ])
    return cleanup_ui, experiments_ui, extra_model_ui, models_ui, run_button, token_ui


@app.cell(hide_code=True)
def _(DEVICE, gc, nfc, time, torch, transformers):
    class Scorer:
        """One Qwen3.5 model. Reads the probability of each answer 0-10 from a single forward pass."""

        def __init__(self, model_id):
            self.model_id = model_id
            t0 = time.time()
            self.tok = transformers.AutoTokenizer.from_pretrained(model_id)
            kw = {"dtype": torch.bfloat16, "output_loading_info": True}
            if DEVICE == "cuda":
                kw["device_map"] = "cuda"
            model, info, errors = None, None, []
            # Qwen3.5 checkpoints are image-text-to-text models; the text part is all that runs here.
            for cls_name in ("AutoModelForImageTextToText", "AutoModelForCausalLM"):
                try:
                    model, info = getattr(transformers, cls_name).from_pretrained(model_id, **kw)
                    break
                except (ValueError, KeyError) as e:
                    errors.append(f"{cls_name}: {e}")
            if model is None:
                raise RuntimeError(f"{model_id} could not be loaded:\n" + "\n".join(errors))
            missing = sorted(info.get("missing_keys") or [])
            if missing:
                raise RuntimeError(f"{model_id}: {len(missing)} weights are missing from the checkpoint "
                                   f"(for example {missing[:3]}), so its scores would be meaningless.")
            self.model = model.eval()
            self.device = next(self.model.parameters()).device
            self.load_seconds = time.time() - t0
            self._answer_tokens()
            self._check_template()

        def _single(self, text):
            ids = self.tok.encode(text, add_special_tokens=False)
            return ids[0] if len(ids) == 1 and self.tok.decode(ids).strip() == text.strip() else None

        def _answer_tokens(self):
            """Token ids that can begin each answer 0-10, with or without a leading space."""
            self.answer_ids = [sorted({i for i in (self._single(str(v)), self._single(" " + str(v))) if i is not None})
                               for v in range(11)]
            self.one_id, self.zero_id = self._single("1"), self._single("0")
            self.split_ten = not self.answer_ids[10]          # "10" is written as "1" then "0"
            if any(not self.answer_ids[v] for v in range(10)) or self.one_id is None or self.zero_id is None:
                raise RuntimeError(f"{self.model_id}: the digits 0-9 are not single tokens, so the expected score "
                                   "cannot be read from one forward pass.")

        def prompt_text(self, prompt):
            msgs = [{"role": "user", "content": prompt}]
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

        def _check_template(self):
            text = self.prompt_text("test")
            template = str(getattr(self.tok, "chat_template", "") or "")
            if "<think>" in template and not text.rstrip().endswith("</think>"):
                raise RuntimeError(f"{self.model_id}: thinking could not be switched off in the chat template.")
            self.template_tail = text[-40:]

        def prompt_ids(self, prompt):
            return self.tok(self.prompt_text(prompt), add_special_tokens=False)["input_ids"]

        def score(self, prompt):
            ids = self.prompt_ids(prompt) + [self.one_id]
            x = torch.tensor([ids], device=self.device)
            with torch.inference_mode():
                out = self.model(input_ids=x, attention_mask=torch.ones_like(x), logits_to_keep=2, use_cache=False)
            logits = out.logits[0, -2:, :].float()
            first = torch.softmax(logits[0], -1)                # the first token of the answer
            mass = [float(first[self.answer_ids[v]].sum()) if self.answer_ids[v] else 0.0 for v in range(11)]
            if self.split_ten:                                   # P("0" after "1") splits the "1" into 1 and 10
                p_zero = float(torch.softmax(logits[1], -1)[self.zero_id])
                mass[10] += mass[1] * p_zero
                mass[1] *= 1 - p_zero
            total = sum(mass)
            probs = [m / total for m in mass] if total > 0 else [float("nan")] * 11
            return {"score": sum(v * p for v, p in enumerate(probs)), "top": max(range(11), key=lambda v: mass[v]),
                    "digit_mass": total, "probs": probs, "prompt_tokens": len(ids) - 1}

        def reply(self, prompt, max_new_tokens=8):
            """What the model actually writes (greedy), as a check that it answers with a number."""
            x = torch.tensor([self.prompt_ids(prompt)], device=self.device)
            with torch.inference_mode():
                out = self.model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=max_new_tokens,
                                          do_sample=False, pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
            return nfc(self.tok.decode(out[0, x.shape[1]:], skip_special_tokens=True)).strip()

        def close(self):
            del self.model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return (Scorer,)


@app.cell(hide_code=True)
def _(HAS_GPU, HF_CACHE, VRAM_GB, free_disk_gb, shutil):
    def repo_size_gb(model_id):
        """Size of the model's weight files on the Hugging Face Hub, or None if it cannot be found out."""
        try:
            from huggingface_hub import HfApi
            info = HfApi().model_info(model_id, files_metadata=True)
            return sum((s.size or 0) for s in info.siblings if s.rfilename.endswith((".safetensors", ".bin"))) / 1e9
        except Exception:
            return None

    def cached_dir(model_id):
        return HF_CACHE / ("models--" + model_id.replace("/", "--"))

    def cached_gb(model_id):
        d = cached_dir(model_id)
        if not d.exists():
            return 0.0
        return sum(f.stat().st_size for f in d.rglob("*") if f.is_file() and not f.is_symlink()) / 1e9

    def delete_cached(model_id):
        shutil.rmtree(cached_dir(model_id), ignore_errors=True)

    def check_room(model_id, others):
        """Make room on disk and check the GPU is big enough. Returns a list of notes; raises RuntimeError if it cannot."""
        notes, need = [], repo_size_gb(model_id)
        if need is None:
            return ["Could not look up the model's size; continuing anyway."]
        if HAS_GPU and need * 1.08 + 2 > VRAM_GB:
            raise RuntimeError(f"{model_id} needs about {need * 1.08 + 2:.0f} GB of GPU memory; this GPU has {VRAM_GB:.0f} GB.")
        if not HAS_GPU and need > 20:
            raise RuntimeError(f"{model_id} ({need:.0f} GB) is too large to run without a GPU.")
        to_fetch = max(0.0, need - cached_gb(model_id))
        if to_fetch + 2 > free_disk_gb():
            for other in others:
                if other != model_id and cached_dir(other).exists():
                    delete_cached(other)
                    notes.append(f"Deleted the downloaded files of {other} to make room.")
                    if to_fetch + 2 <= free_disk_gb():
                        break
        if to_fetch + 2 > free_disk_gb():
            raise RuntimeError(f"{model_id} needs {to_fetch:.0f} GB of disk space for its download; "
                               f"only {free_disk_gb():.0f} GB is free.")
        return notes
    return check_room, delete_cached


@app.cell(hide_code=True)
def _(
    OUT_DIR, Scorer, build_prompt, check_room, cleanup_ui, datetime, delete_cached, experiments_ui, extra_model_ui,
    gpu_problem, json, mo, model_label, models_ui, os, plan, run_button, time, token_ui, transformers,
):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_PATH = OUT_DIR / "window_scores.jsonl"
    MODEL_INFO_PATH = OUT_DIR / "models.json"

    selected_models = list(dict.fromkeys(models_ui.value + ([extra_model_ui.value.strip()] if extra_model_ui.value.strip() else [])))
    selected_experiments = [e for e in plan["experiments"] if e["id"] in experiments_ui.value]
    mo.stop(not selected_models or not selected_experiments,
            mo.callout(mo.md("Choose at least one model and one experiment."), kind="warn"))

    def _key(model_id, e, w):
        return f'{model_id}|{e["fingerprint"]}|{w["window"]}'

    def _load_saved():
        rows = {}
        if RAW_PATH.exists():
            for line in RAW_PATH.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                    rows[r["key"]] = r
                except (json.JSONDecodeError, KeyError):
                    pass
        return rows

    saved = _load_saved()
    _wanted = [(m, e, w) for m in selected_models for e in selected_experiments for w in e["windows"]]
    _missing = [t for t in _wanted if _key(*t) not in saved]
    model_info = json.loads(MODEL_INFO_PATH.read_text()) if MODEL_INFO_PATH.exists() else {}
    run_messages = []

    if not run_button.value:
        mo.stop(len(_missing) == len(_wanted), mo.callout(mo.md(
            f"Press **Run the experiments** to score {len(_wanted)} windows "
            f"({len(selected_experiments)} experiments × {len(selected_models)} models)."), kind="info"))
        if _missing:
            run_messages.append(f"Showing saved scores from an earlier run; {len(_missing)} of {len(_wanted)} windows "
                                "are not scored yet. Press Run to finish them.")
    elif _missing:
        mo.stop(bool(gpu_problem), mo.callout(mo.md(gpu_problem), kind="danger"))
        if token_ui.value.strip():
            os.environ["HF_TOKEN"] = token_ui.value.strip()
        transformers.logging.set_verbosity_error()
        _todo_by_model = {m: [t for t in _missing if t[0] == m] for m in selected_models}
        with mo.status.progress_bar(total=len(_wanted), title="Scoring windows", remove_on_exit=False) as _bar:
            _bar.update(increment=len(_wanted) - len(_missing))
            for _m in selected_models:
                _todo = _todo_by_model[_m]
                if not _todo:
                    continue
                _bar.update(increment=0, title=f"Loading {model_label(_m)}",
                            subtitle="downloading on first use; the 27B model is about 56 GB")
                try:
                    run_messages += check_room(_m, selected_models)
                    _scorer = Scorer(_m)
                except Exception as _e:
                    run_messages.append(f"**{_m} was skipped.** {type(_e).__name__}: {_e}")
                    _bar.update(increment=len(_todo))
                    continue
                _info = {"model": _m, "loaded": datetime.datetime.now().isoformat(timespec="seconds"),
                         "class": type(_scorer.model).__name__, "dtype": str(next(_scorer.model.parameters()).dtype),
                         "device": str(_scorer.device), "load_seconds": round(_scorer.load_seconds, 1),
                         "transformers": transformers.__version__, "ten_is_two_tokens": _scorer.split_ten,
                         "prompt_ending": _scorer.template_tail, "first_replies": {}}
                _first_done = set()
                with open(RAW_PATH, "a", encoding="utf-8") as _f:
                    for _, _e, _w in _todo:
                        _prompt = build_prompt(_e["query_text"], _w["text"])
                        if _e["id"] not in _first_done:      # what the model writes for this experiment's first window
                            _first_done.add(_e["id"])
                            try:
                                _info["first_replies"][_e["id"]] = _scorer.reply(build_prompt(_e["query_text"], _e["windows"][0]["text"]))
                            except Exception as _err:
                                _info["first_replies"][_e["id"]] = f"(could not generate: {_err})"
                        _bar.update(increment=0, title=f"Scoring with {model_label(_m)}",
                                    subtitle=f'{_e["title"]} · window {_w["window"]} of {len(_e["windows"])}')
                        _t0 = time.time()
                        _s = _scorer.score(_prompt)
                        _row = {"key": _key(_m, _e, _w), "model": _m, "experiment_id": _e["id"],
                                "experiment": _e["title"], "fingerprint": _e["fingerprint"], **_w, **_s,
                                "seconds": round(time.time() - _t0, 3),
                                "scored": datetime.datetime.now().isoformat(timespec="seconds")}
                        _f.write(json.dumps(_row, ensure_ascii=False) + "\n")
                        _f.flush()
                        saved[_row["key"]] = _row
                        _bar.update(increment=1)
                _scorer.close()
                model_info[_m] = _info
                MODEL_INFO_PATH.write_text(json.dumps(model_info, ensure_ascii=False, indent=1))
                if cleanup_ui.value:
                    delete_cached(_m)
                    run_messages.append(f"Deleted the downloaded files of {_m}.")

    results_rows = [saved[_key(*t)] for t in _wanted if _key(*t) in saved]
    mo.vstack([mo.callout(mo.md(m), kind="warn" if ("skipped" in m or "not scored" in m) else "neutral")
               for m in run_messages] or [mo.md(f"**{len(results_rows)} of {len(_wanted)} windows scored.**")])
    return model_info, results_rows, selected_experiments, selected_models


@app.cell(hide_code=True)
def _(OUT_DIR, json, mo, model_label, np, pd, plan, results_rows, selected_experiments, selected_models):
    mo.stop(not results_rows)
    models_done = [m for m in selected_models if any(r["model"] == m for r in results_rows)]
    exps_done = [e for e in selected_experiments if any(r["experiment_id"] == e["id"] for r in results_rows)]

    _df = pd.DataFrame(results_rows)
    _df["model_label"] = _df["model"].map(model_label)
    _df["rank"] = (_df.groupby(["model", "experiment_id"])["score"].rank(ascending=False, method="first").astype(int))
    for _v in range(11):
        _df[f"p_{_v}"] = _df["probs"].map(lambda p, v=_v: p[v])
    _order = {e["id"]: i for i, e in enumerate(selected_experiments)}
    _df["_e"] = _df["experiment_id"].map(_order)
    _df["_m"] = _df["model"].map({m: i for i, m in enumerate(selected_models)})
    rankings = (_df.sort_values(["_e", "_m", "rank"])
                .assign(first_word=lambda d: d["start"] + 1, last_word=lambda d: d["end"])
                [["experiment", "model_label", "rank", "window", "references", "score", "top", "digit_mass",
                  *[f"p_{v}" for v in range(11)], "first_word", "last_word", "prompt_tokens", "seconds", "text",
                  "experiment_id", "model"]]
                .rename(columns={"model_label": "model_name", "top": "most_likely_answer",
                                 "digit_mass": "numeric_answer_probability", "text": "window_text"})
                .reset_index(drop=True))

    def word_scores(e, model_id):
        """Each corpus word's score = the mean of the windows that contain it."""
        n = len(e["words"])
        total, count = np.zeros(n), np.zeros(n)
        for r in results_rows:
            if r["model"] == model_id and r["experiment_id"] == e["id"]:
                total[r["start"]:r["end"]] += r["score"]
                count[r["start"]:r["end"]] += 1
        with np.errstate(invalid="ignore"):
            return np.where(count > 0, total / np.maximum(count, 1), np.nan)

    WORD_SCORES = {(e["id"], m): word_scores(e, m) for e in exps_done for m in models_done}

    _word_rows = []
    for _e in exps_done:
        for _i, (_w, _k) in enumerate(zip(_e["words"], _e["word_row"])):
            _row = {"experiment": _e["title"], "word_number": _i + 1, "reference": _e["refs"][_k], "word": _w}
            for _m in models_done:
                _row[model_label(_m)] = round(float(WORD_SCORES[(_e["id"], _m)][_i]), 4)
            _word_rows.append(_row)
    word_table = pd.DataFrame(_word_rows)

    def region_mask(e, kind):
        mask = np.zeros(len(e["words"]), bool)
        rows = np.array(e["word_row"])
        for g in e["regions"]:
            if g["kind"] == kind:
                mask |= (rows >= g["rows"][0]) & (rows < g["rows"][1])
        return mask

    _sum = []
    for _e in exps_done:
        _q, _a, _o = region_mask(_e, "quotation"), region_mask(_e, "allusion"), region_mask(_e, "overlap")
        for _m in models_done:
            _sub = rankings[(rankings["experiment_id"] == _e["id"]) & (rankings["model"] == _m)]
            if _sub.empty:
                continue
            _best = _sub.iloc[0]
            _ws = WORD_SCORES[(_e["id"], _m)]
            _sum.append({
                "experiment": _e["title"], "kind": _e["kind"], "model": model_label(_m),
                "windows scored": f"{len(_sub)} of {len(_e['windows'])}",
                "highest score": round(float(_best["score"]), 2),
                "highest window": f'{int(_best["window"])} ({_best["references"]})',
                "mean score": round(float(_sub["score"].mean()), 2),
                "mean in quoting verses": round(float(np.nanmean(_ws[_q])), 2) if _q.any() else None,
                "mean in proposed-allusion verses": round(float(np.nanmean(_ws[_a])), 2) if _a.any() else None,
                "mean in word-overlap verses": round(float(np.nanmean(_ws[_o])), 2) if _o.any() else None,
                "mean elsewhere": round(float(np.nanmean(_ws[~(_q | _a | _o)])), 2) if (~(_q | _a | _o)).any() else None,
                "numeric-answer probability": round(float(_sub["numeric_answer_probability"].mean()), 3),
                "seconds per window": round(float(_sub["seconds"].mean()), 2),
            })
    summary = pd.DataFrame(_sum)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rankings.drop(columns=["model"]).to_csv(OUT_DIR / "window_rankings.csv", index=False, encoding="utf-8-sig")
    word_table.to_csv(OUT_DIR / "word_scores.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "experiment_file.json").write_text(json.dumps(plan["raw"], ensure_ascii=False, indent=1), encoding="utf-8")

    _tabs = {}
    for _e in exps_done:
        _t = rankings[rankings["experiment_id"] == _e["id"]].drop(columns=["experiment", "experiment_id", "model"])
        _tabs[_e["title"]] = mo.ui.table(_t, selection=None, page_size=15, show_column_summaries=False,
                                         format_mapping={"score": "{:.2f}".format,
                                                         "numeric_answer_probability": "{:.3f}".format,
                                                         **{f"p_{v}": "{:.3f}".format for v in range(11)}})
    mo.vstack([
        mo.md("## 3. Window rankings"),
        mo.md("Rank 1 is the window the model scored highest. *score* is the expected answer, the probability-weighted "
              "mean of 0–10; *p_0*…*p_10* are the probabilities of each answer (rescaled to add up to 1); "
              "*numeric_answer_probability* is how much probability the model put on answering with a number at all."),
        mo.ui.table(summary, selection=None, show_column_summaries=False),
        mo.ui.tabs(_tabs),
    ])
    return WORD_SCORES, exps_done, models_done, rankings, region_mask, summary, word_table


@app.cell(hide_code=True)
def _(MODEL_COLORS, OUT_DIR, WORD_SCORES, common_prefix, exps_done, mo, model_label, models_done, np, short_ref):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
    SHADE = {"quotation": 0.17, "allusion": 0.11, "overlap": 0.06}
    COLOR = {m: MODEL_COLORS[i % len(MODEL_COLORS)] for i, m in enumerate(models_done)}
    _style = {"font.family": "DejaVu Sans", "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
              "ytick.color": MUTED, "text.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
              "savefig.facecolor": SURFACE, "svg.fonttype": "none"}

    def plot_experiment(e, ax):
        n = len(e["words"])
        rows = np.array(e["word_row"])
        starts = [int(np.argmax(rows == k)) for k in range(len(e["refs"])) if (rows == k).any()]
        prefix = common_prefix(e["refs"])
        label_end = [-np.inf, -np.inf]
        for g in sorted(e["regions"], key=lambda g: g["rows"]):     # left to right, so labels can dodge each other
            inside = np.where((rows >= g["rows"][0]) & (rows < g["rows"][1]))[0]
            if not len(inside):
                continue
            a, b = inside[0], inside[-1] + 1
            ax.fill_between([a, b], 0, 10, color=INK, alpha=SHADE[g["kind"]], lw=0)
            if g["label"]:
                half = len(g["label"]) * 0.0042 * n + 0.01 * n       # rough half-width in word units
                c = min(max((a + b) / 2, half), n - half)
                lane = 0 if c - half > label_end[0] else 1 if c - half > label_end[1] else 0
                label_end[lane] = c + half
                ax.text(c, (10.35, 11.3)[lane], g["label"], ha="center", va="bottom", fontsize=7.5, color=INK2)
        xs = np.arange(n) + 0.5
        for m in models_done:
            ax.plot(xs, WORD_SCORES[(e["id"], m)], color=COLOR[m], lw=1.8, solid_capstyle="round",
                    solid_joinstyle="round", label=model_label(m))
        ax.set_xlim(0, n)
        ax.set_ylim(0, 12.4)
        ax.set_yticks(range(0, 11, 2))
        _every = max(1, int(np.ceil(len(starts) / 40)))
        ax.set_xticks(starts[::_every])
        ax.set_xticklabels([short_ref(e["refs"][rows[s]], prefix) for s in starts[::_every]], fontsize=7.5, ha="left")
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_bounds(0, 10)
        ax.set_xlabel(f'{e["corpus"]["name"]}' + (" (verse)" if prefix.endswith(":") else ""), fontsize=9)
        ax.set_ylabel("relatedness (0–10)", fontsize=9)
        _kind = f' ({e["kind"]})' if e["kind"] else ""
        ax.set_title(f'{e["title"]}{_kind}\nquery {e["query"]["name"]} · corpus {e["corpus"]["name"]} · '
                     f'window {e["window_words"]} words, step {e["stride_words"]}', fontsize=10, loc="left", pad=22)

    def legend_handles(e):
        h = [Line2D([], [], color=COLOR[m], lw=1.8, label=model_label(m)) for m in models_done]
        kinds = {g["kind"] for g in e["regions"]}
        if "quotation" in kinds:
            h.append(Patch(color=INK, alpha=SHADE["quotation"], lw=0, label="verses that quote the query"))
        if "allusion" in kinds:
            h.append(Patch(color=INK, alpha=SHADE["allusion"], lw=0, label="proposed allusion or echo"))
        if "overlap" in kinds:
            h.append(Patch(color=INK, alpha=SHADE["overlap"], lw=0, label="shares words with the query, no dependence"))
        return h

    figures = {}
    with plt.rc_context(_style):
        for _e in exps_done:
            _fig, _ax = plt.subplots(figsize=(12, 4.2))
            plot_experiment(_e, _ax)
            _ax.legend(handles=legend_handles(_e), loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False,
                       fontsize=8, labelcolor=INK2)
            _fig.tight_layout()
            for _ext in ("png", "svg"):
                _fig.savefig(OUT_DIR / f"relatedness_{_e['id']}.{_ext}", dpi=160)
            figures[_e["id"]] = _fig
            plt.close(_fig)
        if len(exps_done) > 1:
            _fig, _axes = plt.subplots(len(exps_done), 1, figsize=(12, 3.9 * len(exps_done)), squeeze=False)
            for _ax, _e in zip(_axes[:, 0], exps_done):
                plot_experiment(_e, _ax)
            _fig.legend(handles=legend_handles({"regions": [g for e in exps_done for g in e["regions"]]}),
                        loc="lower center", ncol=4, frameon=False, fontsize=9, labelcolor=INK2)
            _fig.tight_layout(rect=(0, 0.035 * 4 / len(exps_done), 1, 1), h_pad=2.2)
            _fig.savefig(OUT_DIR / "relatedness_all.png", dpi=150)
            plt.close(_fig)

    mo.vstack([
        mo.md("## 4. Relatedness along the corpus"),
        mo.md("Each word gets the mean score of the windows that contain it. Shaded bands mark the verses described "
              "in the experiment file: darkest where the corpus quotes the query, middle where an allusion or echo has been "
              "proposed, lightest where it only shares words with the query."),
        mo.ui.tabs({e["title"]: mo.as_html(figures[e["id"]]) for e in exps_done}),
    ])
    return COLOR, figures


@app.cell(hide_code=True)
def _(COLOR, OUT_DIR, WORD_SCORES, common_prefix, exps_done, html, math, mo, model_label, models_done, short_ref):
    READING_CSS = """
    .swi { font-family: "Gentium Book Plus", "Gentium Plus", "Noto Serif", "DejaVu Serif", "Times New Roman", serif;
           color: #1b1b1a; background: #ffffff; padding: 14px 16px; border-radius: 8px; border: 1px solid #e4e3dc; }
    .swi h3 { font-size: 17px; margin: 0 0 2px; font-weight: 700; }
    .swi .swi-sub { font: 13px/1.4 system-ui, sans-serif; color: #5d5c57; margin: 0 0 10px; }
    .swi .swi-legend { font: 12.5px/1.4 system-ui, sans-serif; color: #3f3e3a; display: flex; flex-wrap: wrap; gap: 6px 16px; margin-bottom: 12px; }
    .swi .swi-key { display: inline-flex; align-items: center; gap: 6px; }
    .swi .swi-key i { display: inline-block; width: 12px; height: 12px; border-radius: 2px; }
    .swi .swi-text { display: flex; flex-wrap: wrap; align-items: flex-end; gap: 12px 7px; }
    .swi .swi-w { display: inline-flex; flex-direction: column; align-items: stretch; }
    .swi .swi-b { display: flex; align-items: flex-end; gap: 1px; height: 32px; background: #f4f3ef;
                  border-bottom: 1px solid #c9c8bf; }
    .swi .swi-b i { display: block; flex: 1 1 0; min-width: 3px; }
    .swi .swi-t { font-size: 18px; line-height: 1.25; padding: 2px 1px 1px; white-space: nowrap; }
    .swi .swi-q .swi-t { background: #dde5f5; }
    .swi .swi-a .swi-t { background: #ebe4f4; }
    .swi .swi-o .swi-t { background: #efede4; }
    .swi .swi-v { font: 10px system-ui, sans-serif; color: #8a8984; vertical-align: super; margin-right: 2px; }
    """

    def reading_view(e):
        prefix = common_prefix(e["refs"])
        kind_of_row, _rank = {}, {"q": 3, "a": 2, "o": 1}
        for g in e["regions"]:
            code = {"quotation": "q", "allusion": "a", "overlap": "o"}[g["kind"]]
            for k in range(*g["rows"]):
                if _rank[code] > _rank.get(kind_of_row.get(k), 0):
                    kind_of_row[k] = code
        parts = []
        for i, (w, k) in enumerate(zip(e["words"], e["word_row"])):
            scores = [(m, WORD_SCORES[(e["id"], m)][i]) for m in models_done]
            bars = "".join(
                f'<i style="height:{0 if math.isnan(s) else max(0.0, min(10.0, s)) * 10:.1f}%;background:{COLOR[m]}"></i>'
                for m, s in scores)
            tip = f'{e["refs"][k]} · ' + " · ".join(f"{model_label(m)} {'–' if math.isnan(s) else f'{s:.1f}'}" for m, s in scores)
            marker = (f'<span class="swi-v">{html.escape(short_ref(e["refs"][k], prefix))}</span>'
                      if i == 0 or e["word_row"][i - 1] != k else "")
            cls = "swi-w" + {"q": " swi-q", "a": " swi-a", "o": " swi-o"}.get(kind_of_row.get(k), "")
            parts.append(f'<span class="{cls}" title="{html.escape(tip)}"><span class="swi-b">{bars}</span>'
                         f'<span class="swi-t">{marker}{html.escape(w)}</span></span>')
        keys = "".join(f'<span class="swi-key"><i style="background:{COLOR[m]}"></i>{html.escape(model_label(m))}</span>'
                       for m in models_done)
        kinds = {g["kind"] for g in e["regions"]}
        if "quotation" in kinds:
            keys += '<span class="swi-key"><i style="background:#dde5f5"></i>verses that quote the query</span>'
        if "allusion" in kinds:
            keys += '<span class="swi-key"><i style="background:#ebe4f4"></i>proposed allusion or echo</span>'
        if "overlap" in kinds:
            keys += '<span class="swi-key"><i style="background:#efede4"></i>shares words with the query, no dependence</span>'
        return (f'<div class="swi"><h3>{html.escape(e["title"])}</h3>'
                f'<p class="swi-sub">Query {html.escape(e["query"]["name"])} · corpus {html.escape(e["corpus"]["name"])} · '
                f'window {e["window_words"]} words, step {e["stride_words"]}. Each bar runs from 0 (empty) to 10 (full) '
                f'and shows the mean score of the windows containing that word. Hover over a word for its scores. '
                f'The small grey numbers are references; the model never saw them.</p>'
                f'<div class="swi-legend">{keys}</div><div class="swi-text">{"".join(parts)}</div></div>')

    reading_html = {e["id"]: reading_view(e) for e in exps_done}
    _page = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
             'content="width=device-width, initial-scale=1"><title>Relatedness above the text</title>'
             '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Gentium+Book+Plus:wght@400;700&display=swap">'
             f'<style>body{{margin:24px;background:#fcfcfb}} .swi{{margin-bottom:28px}} {READING_CSS}</style></head><body>'
             + "".join(reading_html.values()) + "</body></html>")
    (OUT_DIR / "reading_view.html").write_text(_page, encoding="utf-8")

    mo.vstack([
        mo.md("## 5. The corpus with relatedness above each word"),
        mo.ui.tabs({e["title"]: mo.Html(f"<style>{READING_CSS}</style>" + reading_html[e["id"]]) for e in exps_done}),
    ])
    return (reading_html,)


@app.cell(hide_code=True)
def _(OUT_DIR, exps_done, io, json, mo, model_info, models_done, rankings, reading_html, results_rows, summary, zipfile):
    _files = ["window_rankings.csv", "word_scores.csv", "summary.csv", "reading_view.html", "experiment_file.json"]
    _files += [f"relatedness_{e['id']}.{x}" for e in exps_done for x in ("png", "svg")]
    _files += ["relatedness_all.png"] if len(exps_done) > 1 else []

    def _zip_bytes():
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for name in _files:
                if (OUT_DIR / name).exists():
                    z.write(OUT_DIR / name, f"intertextuality_results/{name}")
            z.writestr("intertextuality_results/raw_window_scores.jsonl",
                       "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results_rows))
            z.writestr("intertextuality_results/models.json",
                       json.dumps({m: model_info.get(m, {}) for m in models_done}, ensure_ascii=False, indent=1))
        return buf.getvalue()

    mo.vstack([
        mo.md("## 6. Download"),
        mo.hstack([
            mo.download(data=_zip_bytes, filename="intertextuality_results.zip", mimetype="application/zip",
                        label="Everything (.zip)"),
            mo.download(data=lambda: (OUT_DIR / "window_rankings.csv").read_bytes(), filename="window_rankings.csv",
                        mimetype="text/csv", label="Window rankings (.csv)"),
            mo.download(data=lambda: (OUT_DIR / "reading_view.html").read_bytes(), filename="reading_view.html",
                        mimetype="text/html", label="Reading view (.html)"),
        ], justify="start", gap=1),
        mo.md(
            "**The zip holds**\n\n"
            f"- `window_rankings.csv`: every window with its rank, score, the probability of each answer 0–10, its references and its text ({len(rankings)} rows).\n"
            "- `word_scores.csv`: every corpus word with each model's score (the lines and bars above).\n"
            f"- `summary.csv`: one row per experiment and model ({len(summary)} rows).\n"
            "- `relatedness_<experiment>.png` and `.svg`, and `relatedness_all.png`: the graphs.\n"
            "- `reading_view.html`: the text with bars, to open in any browser.\n"
            "- `raw_window_scores.jsonl`: the scores exactly as saved; `models.json`: how each model was loaded and what it "
            "wrote for the first window of each experiment; `experiment_file.json`: the experiments that were run.\n"
            "\nThe `outputs` folder in the file tree holds the same files, and `window_scores.jsonl` there keeps every "
            "score from this session so that a run can be resumed."),
    ])
    return


@app.cell(hide_code=True)
def _(mo, model_info):
    _rows = "\n".join(
        f"| {m.split('/')[-1]} | {i.get('class', '')} | {i.get('dtype', '')} | {i.get('load_seconds', '')} s | "
        + "; ".join(f"{k}: “{v}”" for k, v in i.get("first_replies", {}).items()) + " |"
        for m, i in model_info.items())
    mo.accordion({
        "How the score is calculated": mo.md(r"""
    The corpus is split into words. A window of the chosen length starts at the first word and moves forward by the step; a final window is added if needed so the last words are covered. For each window the model receives the whole query and the window's text, with no verse numbers, and is asked for a single integer from 0 to 10 (the prompt is in the notebook's code).

    Rather than taking the one number the model would write, the notebook reads the probability the model gives each answer from 0 to 10 in a single forward pass and reports the probability-weighted average. Qwen writes "10" as two tokens, "1" then "0", so the probability of "10" is found by checking how likely "0" is after "1". Thinking is switched off in Qwen3.5's chat template, so the answer starts straight away. Each word's score on the graphs is the average of the windows containing it.

    The models run in bfloat16 on the GPU, one window at a time, with nothing cached between windows.
    """),
        "Models as loaded": mo.md(
            "| model | class | precision | load time | what it wrote for each experiment's first window |\n|---|---|---|---|---|\n"
            + _rows if _rows else "No model has been loaded in this session yet."),
    })
    return


if __name__ == "__main__":
    app.run()
