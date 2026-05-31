"""Lexical detection (Aho-Corasick) + patient-level aggregation and metrics.

Dictionary terms are matched against the normalised document text with an
Aho-Corasick automaton; each match is then checked for negation on the raw
text. A document is positive if it has at least one affirmed term; a patient is
positive if at least one of their documents is positive.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

from .negation import is_negated


@dataclass
class DocPrediction:
    """Document-level prediction with the affirmed / negated terms found."""

    patient_id: str
    label: int
    file_name: str
    prediction: int
    positives: list[str] = field(default_factory=list)
    negated: list[str] = field(default_factory=list)


@dataclass
class PatientPrediction:
    """Patient-level prediction aggregated from document predictions."""

    patient_id: str
    label: int
    prediction: int
    n_docs: int
    n_docs_positive: int


def build_automaton(terms: Sequence[str]):
    """Build an Aho-Corasick automaton from the dictionary terms."""
    import ahocorasick

    unique = sorted(t for t in set(terms) if t.strip())
    if not unique:
        raise ValueError("build_automaton requires at least one non-empty term.")
    automaton = ahocorasick.Automaton()
    for word in unique:
        automaton.add_word(word, word)
    automaton.make_automaton()
    return automaton


def detect(text_norm: str, automaton) -> list[str]:
    """Return dictionary terms present in ``text_norm`` (whole-word matches)."""
    seen: set[str] = set()
    for end_idx, word in automaton.iter(text_norm):
        start_idx = end_idx - len(word) + 1
        left_ok = start_idx == 0 or text_norm[start_idx - 1].isspace()
        right_ok = end_idx == len(text_norm) - 1 or text_norm[end_idx + 1].isspace()
        if left_ok and right_ok:
            seen.add(word)
    return sorted(seen)


def predict_documents(
    records,
    docs_spacy,
    normalized_docs: Sequence[str],
    dictionary_terms: Sequence[str],
    norm_to_raw_forms: dict[str, list[str]],
) -> list[DocPrediction]:
    """Predict each document: positive iff it has >=1 affirmed dictionary term."""
    if len(dictionary_terms) == 0:
        return [
            DocPrediction(rec.patient_id, rec.label, rec.file_name, 0)
            for rec in records
        ]

    automaton = build_automaton(dictionary_terms)
    out: list[DocPrediction] = []
    for rec, doc_spacy, text_norm in zip(records, docs_spacy, normalized_docs):
        positives: list[str] = []
        negated: list[str] = []
        for term_norm in detect(text_norm, automaton):
            raw_forms = norm_to_raw_forms.get(term_norm, [term_norm])
            if is_negated(rec.text, doc_spacy, raw_forms):
                negated.append(term_norm)
            else:
                positives.append(term_norm)
        out.append(
            DocPrediction(
                patient_id=rec.patient_id,
                label=rec.label,
                file_name=rec.file_name,
                prediction=1 if positives else 0,
                positives=positives,
                negated=negated,
            )
        )
    return out


def aggregate_patient_predictions(
    doc_preds: Sequence[DocPrediction],
) -> list[PatientPrediction]:
    """Aggregate document predictions to patient level (any positive doc)."""
    by_patient: dict[tuple[str, int], list[int]] = defaultdict(list)
    for d in doc_preds:
        by_patient[(d.patient_id, d.label)].append(d.prediction)

    out: list[PatientPrediction] = []
    for (patient_id, label), preds in sorted(by_patient.items()):
        n_docs_positive = int(sum(preds))
        out.append(
            PatientPrediction(
                patient_id=patient_id,
                label=label,
                prediction=1 if n_docs_positive > 0 else 0,
                n_docs=len(preds),
                n_docs_positive=n_docs_positive,
            )
        )
    return out


def compute_patient_metrics(
    patient_preds: Sequence[PatientPrediction],
) -> dict[str, float]:
    """Compute TP/TN/FP/FN and precision/recall/F1/specificity/accuracy."""
    tp = sum(1 for p in patient_preds if p.label == 1 and p.prediction == 1)
    tn = sum(1 for p in patient_preds if p.label == 0 and p.prediction == 0)
    fp = sum(1 for p in patient_preds if p.label == 0 and p.prediction == 1)
    fn = sum(1 for p in patient_preds if p.label == 1 and p.prediction == 0)

    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / total if total else 0.0

    return {
        "n_patients": float(total),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "accuracy": accuracy,
    }
