"""Central configuration for the Strategy 1 (E1) asthma detection pipeline.

All tunable hyperparameters, file-system paths and the negation lexicons live
here so that the rest of the package and the results notebook stay free of magic
numbers and hard-coded paths.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Project layout                                                              #
# --------------------------------------------------------------------------- #
# Repository root = two levels up from this file (src/asthma_e1/config.py).
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("ASTHMA_E1_DATA", ROOT_DIR / "data"))
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"


@dataclass(frozen=True)
class Config:
    """Hyperparameters of the reference E1 pipeline.

    Defaults reproduce the configuration reported in the thesis (Section 5.1).
    """

    # spaCy model used for lemmatisation.
    spacy_model: str = "es_core_news_sm"

    # TF-IDF discriminative term selection.
    top_n_terms: int = 5
    min_support: int = 3
    min_ratio: float = 20.0
    tfidf_max_features: int = 500

    # NegEx negation windows (in tokens).
    neg_window: int = 5
    neg_window_strong: int = 10

    # Cross-validation.
    cv_n_splits: int = 5
    random_state: int = 42

    # Patients excluded from evaluation (by id), as in the thesis.
    exclude_patients: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "4580955", "5574419", "4950517", "5586897",
                "85197", "45778", "4365214", "7073838",
                "489996", "7760519", "3289805", "5186875",
            }
        )
    )


@dataclass(frozen=True)
class Paths:
    """File-system locations of the (private) clinical data.

    The clinical data are **not** distributed with this repository. Place your
    own dataset inside ``data/`` (or point ``ASTHMA_E1_DATA`` at another folder)
    following the layout documented in ``data/README.md``.
    """

    data_dir: Path = DATA_DIR
    # Dataset root expected to contain ``asma/`` and ``no_asma/`` subfolders.
    dataset_dir: Path = DATA_DIR / "dataset"
    figures_dir: Path = FIGURES_DIR
    tables_dir: Path = TABLES_DIR


# --------------------------------------------------------------------------- #
# Document filtering                                                          #
# --------------------------------------------------------------------------- #
# Document-type suffixes excluded from the corpus (lab / microbiology reports
# carry little asthma signal and add noise).
EXCLUDED_SUFFIXES: tuple[str, ...] = (
    "_labor", "_micro", "_lab", "_microb", "n2_labor",
)

# Page-footer / boilerplate patterns stripped during cleaning.
FOOTER_PATTERNS: tuple[str, ...] = (
    r"Denominaci[oó] del Centre",
    r"El HCB tratar[aá] sus datos",
    r"L'HCB tractar",
    r"Villarroel,\s*170",
    r"art[ií]culo 9 del Reglamento",
    r"Oficina de Atenci[oó]n a la Ciudadan[ií]a",
    r"pol[ií]tica de protecci[oó]n de datos",
    r"Docum\.\:\s*\d+",
    r"Hist\.\s*Cl[ií]nica\:\s*\d+",
)


# --------------------------------------------------------------------------- #
# Lemma normalisation fixes (spaCy es_core_news_sm)                            #
# --------------------------------------------------------------------------- #
# "asma" (noun) is wrongly lemmatised to "asmo"; Spanish and Catalan adjectival
# forms are normalised to the base noun so all variants map to one search term.
LEMMA_FIX: dict[str, str] = {
    "asmo": "asma",
    "asmático": "asma",
    "asmática": "asma",
    "asmáticos": "asma",
    "asmáticas": "asma",
    "asmàtic": "asma",
    "asmàtica": "asma",
    "asmàtics": "asma",
    "asmàtiques": "asma",
}


# --------------------------------------------------------------------------- #
# NegEx negation cue lexicons (Spanish / Catalan)                             #
# --------------------------------------------------------------------------- #
PRECEDING_NEG: tuple[str, ...] = (
    "no", "sin", "niega", "niegan", "nego", "negó", "ningun", "ninguna",
    "tampoco", "ausencia de", "descarta", "descarto", "descartó",
    "se descarta", "no hay", "no existe", "no presenta", "no refiere",
    "no tiene", "no se observa", "no se aprecia", "ni", "niega sensacion de",
    "sin sensacion de", "no refiere sensacion de", "niega dolor",
    "asintomatico", "asintomatica", "sin clinica de", "descartamos",
    "posible", "posibles", "possible", "possibles", "probable", "probables",
)

FOLLOWING_NEG: tuple[str, ...] = (
    "ausente", "ausentes", "negativo", "negativos", "negativa", "negativas",
    "descartado", "descartada", "no detectado", "negatiu", "negatius",
)

PSEUDO_NEG: tuple[str, ...] = (
    "no solo", "no solamente", "no descarta",
    "no se puede descartar", "no se descarta",
)

TERMINATION: tuple[str, ...] = (
    "pero", "sin embargo", "aunque", "excepto", "salvo",
    "no obstante", "por otro lado", "ademas",
)

STRONG_NEG: tuple[str, ...] = (
    "niega", "niegan", "nego", "negó", "no refiere",
    "no presenta", "no tiene", "no explica", "sin",
)


# Module-level singletons for convenience.
CONFIG = Config()
PATHS = Paths()
