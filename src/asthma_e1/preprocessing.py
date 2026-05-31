"""spaCy lemmatisation and corpus normalisation.

Documents are lemmatised with spaCy (``es_core_news_sm``); stopwords,
punctuation and whitespace are dropped. A small fix map corrects the known
mislemmatisation of "asma" and unifies Spanish/Catalan adjectival variants.
"""

from __future__ import annotations

from collections import defaultdict

from .config import CONFIG, LEMMA_FIX
from .data_loading import DocRecord


def load_spacy_model(model_name: str = CONFIG.spacy_model):
    """Load the spaCy model with parser/NER disabled (lemmatiser only)."""
    import spacy

    try:
        return spacy.load(model_name, disable=["parser", "ner"])
    except Exception:  # pragma: no cover - environment fallback
        return spacy.blank("es")


def build_spacy_docs(records, nlp, batch_size: int = 32) -> list:
    """Run the spaCy pipe over the lowercased document texts."""
    return list(nlp.pipe((r.text.lower() for r in records), batch_size=batch_size))


def _normalised_token(token) -> str | None:
    """Return the normalised lemma for a content token, or ``None`` to drop it."""
    if token.is_punct or token.is_space or token.is_stop:
        return None
    lemma = (token.lemma_ or "").strip().lower()
    lemma = LEMMA_FIX.get(lemma, lemma)
    out = lemma if lemma and lemma != "-pron-" else token.text.lower()
    return out if out and len(out) > 1 else None


def normalize_doc(doc_spacy) -> str:
    """Normalise a single spaCy doc to a space-joined string of lemmas."""
    tokens = [t for t in (_normalised_token(tok) for tok in doc_spacy) if t]
    return " ".join(tokens)


def build_corpus_map(docs_spacy) -> dict[str, list[str]]:
    """Map each normalised lemma to the set of raw surface forms in the corpus.

    Used to expand dictionary terms back to their raw variants for negation
    detection on the original (un-normalised) text.
    """
    cm: dict[str, set[str]] = defaultdict(set)
    for doc in docs_spacy:
        for token in doc:
            out = _normalised_token(token)
            if out:
                cm[out].add(token.text.lower())
    return {k: sorted(v) for k, v in cm.items()}


def build_patient_level_texts(
    records: list[DocRecord], normalized_docs: list[str]
) -> tuple[list[str], list[int], list[str]]:
    """Concatenate per-patient normalised documents.

    Returns ``(patient_texts, patient_labels, patient_ids)`` aligned by index
    and sorted by patient id.
    """
    by_patient: dict[str, list[str]] = defaultdict(list)
    labels: dict[str, int] = {}
    for rec, norm_text in zip(records, normalized_docs):
        by_patient[rec.patient_id].append(norm_text)
        labels[rec.patient_id] = rec.label

    patient_ids = sorted(by_patient)
    patient_texts = [" ".join(by_patient[pid]).strip() for pid in patient_ids]
    patient_labels = [labels[pid] for pid in patient_ids]
    return patient_texts, patient_labels, patient_ids
