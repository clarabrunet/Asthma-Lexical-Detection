"""NegEx-style negation detection adapted to Spanish/Catalan clinical text.

A concept mention is considered *negated* when a negation cue precedes (or
follows) it within a token window, without an intervening termination cue or
sentence break. Pseudo-negations (e.g. "no se puede descartar") affirm the
concept. The implementation mirrors the algorithm reported in the thesis.
"""

from __future__ import annotations

import re
from typing import Sequence

from .config import (
    CONFIG,
    FOLLOWING_NEG,
    PRECEDING_NEG,
    PSEUDO_NEG,
    STRONG_NEG,
    TERMINATION,
)


def _compile(phrases: Sequence[str]) -> list[re.Pattern]:
    """Compile cue phrases into word-boundary regexes, longest first."""
    return [
        re.compile(r"\b" + re.escape(p) + r"\b", re.IGNORECASE)
        for p in sorted(set(phrases), key=len, reverse=True)
    ]


PRE_PAT = _compile(PRECEDING_NEG)
POST_PAT = _compile(FOLLOWING_NEG)
PSEUDO_PAT = _compile(PSEUDO_NEG)
TERM_PAT = _compile(TERMINATION)
STRONG_PAT = _compile(STRONG_NEG)


def _find_all(patterns: Sequence[re.Pattern], text: str) -> list[tuple[int, int]]:
    """Return sorted (start, end) char spans of all pattern matches."""
    spans: list[tuple[int, int]] = []
    for pat in patterns:
        spans.extend((m.start(), m.end()) for m in pat.finditer(text))
    spans.sort()
    return spans


def is_negated(
    text: str,
    doc_spacy,
    concepts: Sequence[str],
    window: int = CONFIG.neg_window,
    window_strong: int = CONFIG.neg_window_strong,
) -> bool:
    """Whether *all* mentions of ``concepts`` in ``text`` are negated.

    Returns ``True`` only if at least one concept mention is found and every
    occurrence is negated. A single affirmed (or pseudo-negated) occurrence
    makes the concept count as present and returns ``False``.

    Parameters
    ----------
    text:
        Raw (un-normalised) document text.
    doc_spacy:
        The spaCy doc for ``text.lower()`` (used for token-distance windows).
    concepts:
        Raw surface forms of the concept to check.
    window:
        Token window for ordinary preceding/following cues.
    window_strong:
        Wider token window for strong negation cues.
    """
    text_lower = text.lower()
    pseudo_spans = _find_all(PSEUDO_PAT, text_lower)
    term_spans = _find_all(TERM_PAT, text_lower)
    pre_spans = _find_all(PRE_PAT, text_lower)
    strong_spans = _find_all(STRONG_PAT, text_lower)
    post_spans = _find_all(POST_PAT, text_lower)

    # Precompute char-position -> token-index for fast distance checks.
    char_to_tok: dict[int, int] = {}
    for i, tok in enumerate(doc_spacy):
        for c in range(tok.idx, tok.idx + len(tok)):
            char_to_tok[c] = i

    def _ti(char_idx: int) -> int:
        return char_to_tok.get(char_idx, -1)

    found_any = False
    for concept in concepts:
        concept_l = str(concept).lower().strip()
        if not concept_l:
            continue
        for m in re.finditer(r"\b" + re.escape(concept_l) + r"\b", text_lower):
            found_any = True
            c_start, c_end = m.start(), m.end()
            c_tok = _ti(c_start)
            if c_tok == -1:
                continue

            # Pseudo-negation surrounding the concept -> affirmed.
            if any(ps <= c_start and c_end <= pe for ps, pe in pseudo_spans):
                return False

            this_neg = False

            # Ordinary preceding negation (comma also acts as a break).
            for t_start, t_end in pre_spans:
                if t_end > c_start:
                    continue
                t_tok = _ti(t_start)
                if t_tok == -1 or c_tok - t_tok > window:
                    continue
                if re.search(r"[\.;,]", text_lower[t_end:c_start]):
                    continue
                if not any(t_end <= ts <= c_start for ts, _ in term_spans):
                    this_neg = True
                    break

            # Strong negation with a wider window.
            if not this_neg:
                for t_start, t_end in strong_spans:
                    if t_end > c_start:
                        continue
                    t_tok = _ti(t_start)
                    if t_tok == -1 or c_tok - t_tok > window_strong:
                        continue
                    if re.search(r"[\.;,]", text_lower[t_end:c_start]):
                        continue
                    if not any(t_end <= ts <= c_start for ts, _ in term_spans):
                        this_neg = True
                        break

            # Following negation.
            if not this_neg:
                c_end_tok = _ti(max(c_start, c_end - 1))
                if c_end_tok == -1:
                    c_end_tok = c_tok
                for t_start, _ in post_spans:
                    if t_start < c_end:
                        continue
                    g_end = _ti(t_start)
                    if g_end == -1:
                        g_end = len(doc_spacy) - 1
                    if g_end - c_end_tok > window:
                        continue
                    if not any(c_end <= ts <= t_start for ts, _ in term_spans):
                        this_neg = True
                        break

            # An affirmed occurrence means the concept is present.
            if not this_neg:
                return False

    # True only if mentions were found and all were negated.
    return found_any
