# Asthma-Lexical-Detection

### Lexical rule-based detection of active asthma in Spanish clinical notes

> Strategy 1 (**E1**) of the Master's thesis — an interpretable screening
> pipeline that detects **active asthma** from free-text Spanish electronic
> health records (EHRs) using a small discriminative dictionary and negation
> detection.

---

## Table of contents

- [Overview](#overview)
- [Method](#method)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
- [Reproducing the thesis results](#reproducing-the-thesis-results)
- [Configuration](#configuration)
- [Testing](#testing)
- [Citation](#citation)
- [License](#license)

---

## Overview

**Asthma-Lexical-Detection** identifies patients with active asthma from clinical
narratives using a fully **interpretable, rule-based** approach: a compact
dictionary of discriminative terms, exact lexical matching, and clinical
negation detection. Every decision is traceable to the exact term and sentence
that triggered it — a key property for clinical screening and epidemiological
auditing.

The dictionary is learned per cross-validation fold from the TF-IDF
asthma/non-asthma ratio, so no opaque classifier is involved. This is the
companion **baseline (Strategy 1)** to the embedding-based system
(Strategy 2, *Semantic-Modeling*).

## Method

The pipeline runs five interpretable stages and is evaluated with patient-level
stratified cross-validation:

| # | Stage | Module |
|---|-------|--------|
| 1 | **Cleaning** — strip page footers, boilerplate and uninformative lines; exclude lab/microbiology reports. | `data_loading.py` |
| 2 | **Lemmatisation** — spaCy `es_core_news_sm`, with a fix map for "asma" and its Spanish/Catalan adjectival variants. | `preprocessing.py` |
| 3 | **Dictionary selection** — top-N terms by TF-IDF asthma/non-asthma ratio (min ratio, min support), re-selected per training fold to avoid leakage. | `lexicon.py` |
| 4 | **Lexical matching** — Aho-Corasick whole-word search of dictionary terms. | `detection.py` |
| 5 | **Negation (NegEx)** — discard negated mentions with Spanish/Catalan cue lexicons (preceding/following/strong/pseudo cues + termination). | `negation.py` |

A document is positive if it contains at least one **affirmed** dictionary term;
a patient is positive if at least one of their documents is positive
(`detection.py`). Evaluation is **stratified 5-fold CV** at the patient level
(`evaluation.py`).

```
clinical notes ─► clean ─► spaCy lemmas ─┬─► TF-IDF dictionary (per fold)
                                         │            │
                                         └─► Aho-Corasick match ─► NegEx filter
                                                      │
                              affirmed term? ─► document positive ─► patient positive
```

## Repository layout

```
Asthma-Lexical-Detection/
├── src/asthma_e1/            # the importable package
│   ├── config.py             # hyperparameters, paths, cue lexicons
│   ├── data_loading.py       # corpus loading + document cleaning
│   ├── preprocessing.py      # spaCy lemmatisation + normalisation
│   ├── lexicon.py            # TF-IDF discriminative dictionary
│   ├── negation.py           # NegEx negation engine
│   ├── detection.py          # Aho-Corasick + aggregation + metrics
│   └── evaluation.py         # stratified k-fold CV
├── scripts/
│   └── run_pipeline.py       # end-to-end CV from the command line
├── notebooks/
│   └── results.ipynb         # narrative results (thesis Sections 5.1–5.2)
├── tests/                    # unit tests on synthetic data (no clinical data)
├── data/                     # clinical corpus — empty, see data/README.md
└── results/                  # generated figures/ and tables/
```

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
python -m spacy download es_core_news_sm   # if not pulled in automatically
```

## Data

The clinical data used in the thesis are **confidential** and are **not**
included — `data/` ships empty. The pipeline expects a
`data/dataset/{asma,no_asma}/<patient_id>/*.txt` tree; see
**[`data/README.md`](data/README.md)** for the exact layout.

## Usage

End-to-end cross-validation from the command line:

```bash
python scripts/run_pipeline.py --dataset data/dataset --top_n 5 --n_splits 5
```

Programmatic use:

```python
from asthma_e1 import (
    load_dataset_records, load_spacy_model, build_spacy_docs,
    normalize_doc, build_corpus_map, run_cv, CONFIG,
)

records   = load_dataset_records("data/dataset", CONFIG.exclude_patients)
nlp       = load_spacy_model()
docs      = build_spacy_docs(records, nlp)
norm_docs = [normalize_doc(d) for d in docs]
corpus    = build_corpus_map(docs)

metrics, oof = run_cv(records, docs, norm_docs, corpus, top_n=5, n_splits=5)
print(metrics)
```

Or work through **[`notebooks/results.ipynb`](notebooks/results.ipynb)**, which
reproduces the corpus description, dictionary analysis, CV metrics, NegEx
behaviour, error analysis and hyperparameter sensitivity reported in the thesis.

## Reproducing the thesis results

`notebooks/results.ipynb` maps onto the thesis:

| Notebook section | Thesis | Output |
|------------------|--------|--------|
| Corpus description | 5.1 | `corpus_summary.csv`, `fig_corpus_overview.png` |
| Discriminative dictionary | 5.1 | `dictionary_stability.csv`, `fig_dictionary.png` |
| Cross-validation metrics | 5.2.3 | `metrics_cv_summary.csv`, `fig_confusion_matrix.png` |
| NegEx behaviour | 5.2 | `negex_per_term.csv`, `fig_negex_per_term.png` |
| Error analysis (FN / FP) | 5.2.4 | `errors_FN.csv`, `errors_FP.csv` |
| Hyperparameter sensitivity | 5.2 | `sensitivity_*.csv`, `fig_sensitivity.png` |
| Computational cost | 5.2 | `timing_per_fold.csv` |

## Configuration

All hyperparameters live in `src/asthma_e1/config.py` and reproduce the thesis
defaults:

| Group | Parameter | Default |
|-------|-----------|---------|
| Lemmatiser | `spacy_model` | `es_core_news_sm` |
| Dictionary | `top_n_terms` | 5 |
| | `min_support` | 3 |
| | `min_ratio` | 20.0 |
| NegEx | `neg_window` | 5 |
| | `neg_window_strong` | 10 |
| CV | `cv_n_splits` | 5 |
| | `random_state` | 42 |

Point the pipeline at a different data folder via the `ASTHMA_E1_DATA`
environment variable.

## Testing

```bash
pytest
```

Tests run on small synthetic inputs and require neither the clinical data nor a
trained spaCy model (a blank tokenizer is used).

## Citation

> Clara Brunet. *Strategy 1 (E1): lexical rule-based asthma detection in Spanish
> clinical notes.* Master's thesis, 2026.

## License

[MIT](LICENSE) — **source code only**. The clinical data are confidential and
are not covered by this license.
