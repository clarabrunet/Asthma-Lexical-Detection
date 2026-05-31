"""Tests for the NegEx negation engine (uses a blank spaCy tokenizer)."""

import pytest

spacy = pytest.importorskip("spacy")

from asthma_e1.negation import is_negated


@pytest.fixture(scope="module")
def nlp():
    return spacy.blank("es")


def _doc(nlp, text):
    return nlp(text.lower())


def test_affirmed_mention_is_not_negated(nlp):
    text = "El paciente presenta asma bronquial."
    assert is_negated(text, _doc(nlp, text), ["asma"]) is False


def test_preceding_negation(nlp):
    text = "No presenta asma en el momento actual."
    assert is_negated(text, _doc(nlp, text), ["asma"]) is True


def test_pseudo_negation_affirms(nlp):
    text = "No se puede descartar asma bronquial."
    assert is_negated(text, _doc(nlp, text), ["asma"]) is False


def test_returns_false_when_concept_absent(nlp):
    text = "El paciente presenta hipertension arterial."
    assert is_negated(text, _doc(nlp, text), ["asma"]) is False


def test_one_affirmed_among_negated_counts_as_present(nlp):
    text = "No presenta asma. Mas adelante se confirma asma bronquial."
    # Second mention is affirmed -> concept present -> not negated overall.
    assert is_negated(text, _doc(nlp, text), ["asma"]) is False
