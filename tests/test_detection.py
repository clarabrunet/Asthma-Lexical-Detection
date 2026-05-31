"""Tests for lexical detection, patient aggregation and metrics."""

import pytest

from asthma_e1.detection import (
    DocPrediction,
    aggregate_patient_predictions,
    compute_patient_metrics,
    detect,
)

pytest.importorskip("ahocorasick")
from asthma_e1.detection import build_automaton  # noqa: E402


def test_detect_whole_word_matches():
    automaton = build_automaton(["asma", "disnea"])
    found = detect("paciente con asma y disnea leve", automaton)
    assert found == ["asma", "disnea"]
    # Substring inside another word must not match.
    assert detect("asmatico sin diagnostico", automaton) == []


def test_aggregate_any_positive_doc_makes_patient_positive():
    doc_preds = [
        DocPrediction("p1", 1, "a.txt", 0),
        DocPrediction("p1", 1, "b.txt", 1),
        DocPrediction("p2", 0, "c.txt", 0),
    ]
    pats = {p.patient_id: p for p in aggregate_patient_predictions(doc_preds)}
    assert pats["p1"].prediction == 1
    assert pats["p1"].n_docs == 2 and pats["p1"].n_docs_positive == 1
    assert pats["p2"].prediction == 0


def test_compute_patient_metrics_perfect():
    from asthma_e1.detection import PatientPrediction

    preds = [
        PatientPrediction("p1", 1, 1, 1, 1),
        PatientPrediction("p2", 0, 0, 1, 0),
    ]
    m = compute_patient_metrics(preds)
    assert m["precision"] == 1.0 and m["recall"] == 1.0
    assert m["f1"] == 1.0 and m["accuracy"] == 1.0
