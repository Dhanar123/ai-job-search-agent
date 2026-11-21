"""Text normalization service for company names.

Normalizes case, whitespace, punctuation, and legal suffixes to produce
comparison keys without ever mutating the original company name.
"""

from __future__ import annotations

import re
import unicodedata

# Legal entity suffixes to strip from the end of normalized company keys
# Ordered from multi-word phrases to single words
_LEGAL_SUFFIXES = [
    r"\bprivate\s+limited\b",
    r"\bpvt\s+ltd\b",
    r"\bpvt\b",
    r"\bpty\s+ltd\b",
    r"\bltd\b",
    r"\blimited\b",
    r"\bincorporated\b",
    r"\binc\b",
    r"\bcorporation\b",
    r"\bcorp\b",
    r"\btechnologies\b",
    r"\btechnology\b",
    r"\bllc\b",
    r"\bllp\b",
    r"\bgmbh\b",
    r"\bplc\b",
    r"\bcompany\b",
    r"\bco\b",
]

_PUNCTUATION_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_company_name(name: str) -> str:
    """Produce a normalized comparison key for a company name.

    Rules:
    - Unicode-normalize (NFKC).
    - Lowercase.
    - Remove punctuation (replace with single space or drop).
    - Strip legal suffixes repeatedly from the end.
    - Strip leading or trailing 'the'.
    - Collapse and trim whitespace.
    - Never mutates the original string.

    Args:
        name: Raw original company name.

    Returns:
        Clean normalized key for deduplication and matching.
    """
    if not name:
        return ""

    # NFKC Unicode normalization
    text = unicodedata.normalize("NFKC", name)
    text = text.lower()

    # Replace punctuation with spaces
    text = _PUNCTUATION_RE.sub(" ", text)

    # Collapse whitespace
    text = _WHITESPACE_RE.sub(" ", text).strip()

    # Iteratively strip trailing legal suffixes and trailing/leading "the"
    changed = True
    while changed:
        changed = False

        # Drop trailing or leading 'the'
        if text.endswith(" the"):
            text = text[:-4].strip()
            changed = True
        if text.startswith("the "):
            text = text[4:].strip()
            changed = True

        for suffix in _LEGAL_SUFFIXES:
            pattern = re.compile(suffix + r"$", flags=re.IGNORECASE)
            match = pattern.search(text)
            if match:
                text = text[: match.start()].strip()
                changed = True
                break

    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text
