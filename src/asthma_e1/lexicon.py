"""Discriminative dictionary selection via TF-IDF asthma/non-asthma ratio.

Terms are ranked by the ratio of their mean TF-IDF weight in asthma vs.
non-asthma patients. Only terms whose ratio exceeds ``min_ratio`` and that are
supported by at least ``min_support`` positive patients are kept, up to
``top_n`` terms. Selecting the dictionary per training fold avoids leakage.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import CONFIG


def select_tfidf_terms(
    texts: list[str],
    labels: list[int],
    top_n: int = CONFIG.top_n_terms,
    min_support: int = CONFIG.min_support,
    max_features: int = CONFIG.tfidf_max_features,
    min_ratio: float = CONFIG.min_ratio,
) -> list[str]:
    """Select up to ``top_n`` discriminative terms by TF-IDF ratio.

    Parameters
    ----------
    texts, labels:
        Patient-level normalised texts and binary labels (1 = asthma).
    top_n:
        Maximum number of terms to return.
    min_support:
        Minimum number of positive patients a term must appear in.
    min_ratio:
        Minimum asthma/non-asthma mean-TF-IDF ratio; terms below this are not
        selected (the ranking is descending, so iteration stops at the first
        term below the threshold).

    Returns
    -------
    list[str]
        Selected dictionary terms, most discriminative first.
    """
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 1))
    matrix = vec.fit_transform(texts)
    feats = np.array(vec.get_feature_names_out())
    y = np.array(labels)
    pos_mask, neg_mask = y == 1, y == 0
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return []

    mean_pos = np.asarray(matrix[pos_mask].mean(axis=0)).ravel()
    mean_neg = np.asarray(matrix[neg_mask].mean(axis=0)).ravel()
    ratio = (mean_pos + 1e-9) / (mean_neg + 1e-9)
    support = np.asarray((matrix[pos_mask] > 0).sum(axis=0)).ravel()

    selected: list[str] = []
    for idx in np.argsort(-ratio):
        term = str(feats[idx])
        if support[idx] < min_support or len(term.strip()) <= 2:
            continue
        if ratio[idx] < min_ratio:
            break
        selected.append(term)
        if len(selected) >= top_n:
            break
    return selected


def build_norm_to_raw(
    dictionary: list[str], corpus_map: dict[str, list[str]]
) -> dict[str, list[str]]:
    """Map each dictionary term to its raw surface forms in the corpus."""
    return {
        term: sorted(set(corpus_map.get(term, [term])))
        for term in dictionary
    }
