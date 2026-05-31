"""Loading and cleaning of the (private) clinical corpus.

The dataset is expected under ``data/dataset/{asma,no_asma}/<patient_id>/*.txt``.
The clinical data are confidential and are **not** shipped with the repository;
see ``data/README.md``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .config import EXCLUDED_SUFFIXES, FOOTER_PATTERNS, PATHS

_FOOTER_RE = [re.compile(p, re.IGNORECASE) for p in FOOTER_PATTERNS]
_DATE_HEADER_RE = re.compile(r"^Fecha:\s*\d{2}\.\d{2}\.\d{4}")
_SECTION_RULE_RE = re.compile(r"^_{5,}$")
_ONLY_SEPARATORS_RE = re.compile(r"^[\s\-\_\=]+$")
_ONLY_NUMBERS_RE = re.compile(r"^[\d\s\.\:\-\/\(\)\%\,]+$")


@dataclass
class DocRecord:
    """A single clinical document belonging to a patient."""

    patient_id: str
    label: int  # 1 = active asthma, 0 = non-asthma
    file_name: str
    text: str


def safe_read_text(path: Path) -> str:
    """Read a text file trying several encodings (utf-8, cp1252, latin1)."""
    for enc in ("utf-8", "cp1252", "latin1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="latin1", errors="ignore")


def is_excluded(path: Path, excluded_suffixes=EXCLUDED_SUFFIXES) -> bool:
    """Whether a file is an excluded document type (lab / microbiology)."""
    stem = path.stem.lower()
    return any(stem.endswith(suffix) for suffix in excluded_suffixes)


def clean_document(text: str) -> str:
    """Strip page footers, boilerplate and uninformative lines.

    A footer pattern starts a *skip* region that is reset at the next clinical
    section header (``Fecha: dd.mm.yyyy``) or horizontal rule. Empty, very short
    (< 10 chars), separator-only and number-only lines are dropped.
    """
    out: list[str] = []
    skip = False
    for raw_line in text.splitlines():
        s = raw_line.strip()
        if _DATE_HEADER_RE.match(s):  # new clinical section -> stop skipping
            skip = False
        if any(p.search(s) for p in _FOOTER_RE):
            skip = True
            continue
        if _SECTION_RULE_RE.match(s):
            skip = False
            continue
        if skip or not s:
            continue
        if _ONLY_SEPARATORS_RE.match(s):
            continue
        if _ONLY_NUMBERS_RE.match(s):
            continue
        if len(s) < 10:
            continue
        out.append(s)
    return " ".join(out)


def load_dataset_records(
    dataset_root: Path = PATHS.dataset_dir,
    exclude_patients: frozenset[str] | None = None,
) -> list[DocRecord]:
    """Load and clean all documents from a ``{asma,no_asma}`` dataset tree.

    Parameters
    ----------
    dataset_root:
        Folder containing ``asma/`` and ``no_asma/`` subfolders, each with one
        subfolder per patient.
    exclude_patients:
        Optional set of patient ids to drop entirely.

    Returns
    -------
    list[DocRecord]
        One record per (non-empty, non-excluded) document.
    """
    exclude_patients = exclude_patients or frozenset()
    asma_root = dataset_root / "asma"
    no_asma_root = dataset_root / "no_asma"
    if not asma_root.exists() or not no_asma_root.exists():
        raise FileNotFoundError(
            f"Expected '{dataset_root}/asma' and '{dataset_root}/no_asma'. "
            "See data/README.md for the required layout."
        )

    records: list[DocRecord] = []
    for folder, label in ((asma_root, 1), (no_asma_root, 0)):
        for patient_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
            if patient_dir.name in exclude_patients:
                continue
            for txt_file in sorted(patient_dir.glob("*.txt")):
                if is_excluded(txt_file):
                    continue
                text = clean_document(safe_read_text(txt_file))
                if not text.strip():
                    continue
                records.append(
                    DocRecord(
                        patient_id=patient_dir.name,
                        label=label,
                        file_name=txt_file.name,
                        text=text,
                    )
                )
    return records
