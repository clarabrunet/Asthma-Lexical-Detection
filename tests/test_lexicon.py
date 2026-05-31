"""Tests for TF-IDF discriminative dictionary selection."""

from asthma_e1.lexicon import build_norm_to_raw, select_tfidf_terms


def test_selects_discriminative_term():
    # "asma" appears only in positive patients; "control" in both.
    texts = [
        "asma disnea control",
        "asma sibilancias control",
        "asma tos control",
        "control rutina visita",
        "control rutina analitica",
        "control rutina seguimiento",
    ]
    labels = [1, 1, 1, 0, 0, 0]
    terms = select_tfidf_terms(texts, labels, top_n=5, min_support=3, min_ratio=2.0)
    assert "asma" in terms
    assert "control" not in terms


def test_min_ratio_can_empty_dictionary():
    texts = ["asma control", "asma control", "control asma", "control asma"]
    labels = [1, 1, 0, 0]
    # Nothing is discriminative enough at a very high ratio threshold.
    assert select_tfidf_terms(texts, labels, min_ratio=100.0) == []


def test_build_norm_to_raw_falls_back_to_term():
    corpus_map = {"asma": ["asma", "asmatica"]}
    out = build_norm_to_raw(["asma", "disnea"], corpus_map)
    assert out["asma"] == ["asma", "asmatica"]
    assert out["disnea"] == ["disnea"]  # not in map -> itself
