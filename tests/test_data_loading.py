"""Tests for document cleaning and file filtering."""

from pathlib import Path

from asthma_e1.data_loading import clean_document, is_excluded


def test_clean_document_drops_footer_and_short_lines():
    raw = (
        "El paciente presenta asma bronquial y tos persistente.\n"
        "___\n"
        "Denominació del Centre: Hospital X\n"
        "Villarroel, 170\n"
        "Fecha: 01.02.2023\n"
        "Refiere disnea de esfuerzo desde hace meses.\n"
        "123 456\n"
        "ok\n"
    )
    cleaned = clean_document(raw)
    assert "asma bronquial" in cleaned
    assert "disnea de esfuerzo" in cleaned
    # Footer line and the boilerplate after it are removed.
    assert "Villarroel" not in cleaned
    assert "Denominaci" not in cleaned
    # Pure-number and too-short lines are dropped.
    assert "123 456" not in cleaned
    assert " ok" not in cleaned


def test_is_excluded_matches_lab_suffixes():
    assert is_excluded(Path("12345_n2_labor.txt"))
    assert is_excluded(Path("9_microb.txt"))
    assert not is_excluded(Path("9_nota_clin.txt"))
