# Asthma-Lexical-Detection

Asthma-Lexical-Detection is  interpretable screening pipeline that detects active asthma from free-text Spanish electronic
health records (EHRs) using a small discriminative dictionary and negation
detection.

---


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
a patient is positive if at least one of their documents is positive. Evaluation is **stratified 5-fold CV** at the patient level.

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
included 
