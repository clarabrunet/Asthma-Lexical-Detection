"""Stratified k-fold cross-validation at the patient level.

The dictionary is re-selected on each training fold (avoiding leakage), then
applied to the held-out patients. Out-of-fold patient predictions are pooled
for global metrics.
"""

from __future__ import annotations

from collections import defaultdict

from sklearn.model_selection import StratifiedKFold

from .config import CONFIG
from .detection import (
    aggregate_patient_predictions,
    compute_patient_metrics,
    predict_documents,
)
from .lexicon import build_norm_to_raw, select_tfidf_terms
from .preprocessing import build_patient_level_texts


def run_cv(
    records,
    docs_spacy,
    norm_docs,
    corpus_map,
    top_n: int = CONFIG.top_n_terms,
    n_splits: int = CONFIG.cv_n_splits,
    random_state: int = CONFIG.random_state,
) -> tuple[dict[str, float], list]:
    """Run patient-level stratified CV and return pooled OOF metrics.

    Returns ``(metrics, oof_patient_predictions)`` where ``metrics`` are the
    pooled out-of-fold patient metrics.
    """
    pat_texts, pat_labels, pat_ids = build_patient_level_texts(records, norm_docs)
    pat_text_by_id = dict(zip(pat_ids, pat_texts))
    pat_label_by_id = dict(zip(pat_ids, pat_labels))

    pat_to_idx: dict[str, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        pat_to_idx[rec.patient_id].append(i)

    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    oof_preds = []
    for train_idx, test_idx in skf.split(pat_ids, pat_labels):
        train_ids = [pat_ids[i] for i in train_idx]
        test_ids = [pat_ids[i] for i in test_idx]

        train_texts = [pat_text_by_id[pid] for pid in train_ids]
        train_labels = [pat_label_by_id[pid] for pid in train_ids]

        test_doc_idx = sorted(i for pid in test_ids for i in pat_to_idx[pid])
        test_records = [records[i] for i in test_doc_idx]
        test_docs = [docs_spacy[i] for i in test_doc_idx]
        test_norm = [norm_docs[i] for i in test_doc_idx]

        dictionary = select_tfidf_terms(train_texts, train_labels, top_n=top_n)
        norm_to_raw = build_norm_to_raw(dictionary, corpus_map)

        doc_preds = predict_documents(
            records=test_records,
            docs_spacy=test_docs,
            normalized_docs=test_norm,
            dictionary_terms=dictionary,
            norm_to_raw_forms=norm_to_raw,
        )
        oof_preds.extend(aggregate_patient_predictions(doc_preds))

    return compute_patient_metrics(oof_preds), oof_preds
