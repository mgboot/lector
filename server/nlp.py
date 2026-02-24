"""Shared LatinCy (spaCy) model loader — singleton to avoid loading twice."""

import spacy

_nlp = None


def get_nlp():
    """Lazy-load and return the LatinCy spaCy model."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("la_core_web_lg")
        _nlp.max_length = 2_500_000
    return _nlp
