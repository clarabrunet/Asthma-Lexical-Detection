"""Strategy 1 (E1): lexical rule-based asthma detection in Spanish EHRs.

An interpretable screening pipeline that detects active asthma from free-text
Spanish clinical notes by:

1. cleaning and lemmatising documents with spaCy;
2. selecting a small discriminative dictionary by TF-IDF asthma/non-asthma ratio
   (re-selected per CV fold to avoid leakage);
3. matching terms with an Aho-Corasick automaton;
4. discarding negated mentions with an adapted NegEx engine;
5. aggregating to patient level and evaluating with stratified k-fold CV.

See the module docstrings and ``notebooks/results.ipynb`` for details.
"""

from __future__ import annotations

from .config import CONFIG, PATHS, Config, Paths
from .data_loading import DocRecord, clean_document, load_dataset_records
from .detection import (
    aggregate_patient_predictions,
    compute_patient_metrics,
    detect,
    predict_documents,
)
from .evaluation import run_cv
from .lexicon import build_norm_to_raw, select_tfidf_terms
from .negation import is_negated
from .preprocessing import (
    build_corpus_map,
    build_patient_level_texts,
    build_spacy_docs,
    load_spacy_model,
    normalize_doc,
)

__all__ = [
    "CONFIG",
    "PATHS",
    "Config",
    "Paths",
    "DocRecord",
    "clean_document",
    "load_dataset_records",
    "load_spacy_model",
    "build_spacy_docs",
    "normalize_doc",
    "build_corpus_map",
    "build_patient_level_texts",
    "select_tfidf_terms",
    "build_norm_to_raw",
    "is_negated",
    "detect",
    "predict_documents",
    "aggregate_patient_predictions",
    "compute_patient_metrics",
    "run_cv",
]

__version__ = "1.0.0"
