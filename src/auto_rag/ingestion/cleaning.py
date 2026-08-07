"""Text cleaning.

Normalises raw extracted text so chunking and embedding operate on
consistent, readable content: hyphenated line breaks are rejoined, exotic
whitespace is collapsed, and page-number/footer artifacts are dropped.
"""

from __future__ import annotations

import re
import unicodedata

_PAGE_NUMBER_LINE = re.compile(r"^\s*[-–—|·]\s*\d+\s*[-–—|·]?\s*$")
_HYPHENATED_BREAK = re.compile(r"(?<=[a-z])-\s*\r?\n\s*(?=[a-z])")
_SOFT_HYPHEN = re.compile(r"\u00ad")
_WHITESPACE_RUN = re.compile(r"[ \t]+")
_NEWLINE_RUN = re.compile(r"\n{3,}")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(text: str) -> str:
    """Clean raw extracted text for embedding and chunking.

    Args:
        text: Raw text from a loader.

    Returns:
        Normalised, human-readable text.
    """
    if not text:
        return ""

    # Unicode normalisation and whitespace unification.
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")
    text = _CONTROL_CHARS.sub("", text)

    # Rejoin hyphenated line breaks ("manu-\nal" -> "manual").
    text = _HYPHENATED_BREAK.sub("", text)
    text = _SOFT_HYPHEN.sub("", text)

    # Collapse horizontal whitespace and excessive blank lines.
    text = _WHITESPACE_RUN.sub(" ", text)
    text = _NEWLINE_RUN.sub("\n\n", text)

    # Drop standalone page-number / footer lines.
    lines = [line for line in text.splitlines() if not _PAGE_NUMBER_LINE.match(line)]
    return "\n".join(lines).strip()
