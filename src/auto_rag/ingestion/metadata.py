"""Metadata extraction.

Infers document type, vehicle applicability (make / model / year / engine /
VIN), and title from the filename and the first pages of a document. This
metadata is stored per chunk in the vector store and used for filtering in
retrieval (Phase 4).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from auto_rag.constants import (
    DOCUMENT_TYPE_DTC,
    DOCUMENT_TYPE_REPAIR_MANUAL,
    DOCUMENT_TYPE_SERVICE_MANUAL,
    DOCUMENT_TYPE_TSB,
    DOCUMENT_TYPE_WIRING_DIAGRAM,
)
from auto_rag.db.models import DocType
from auto_rag.ingestion.loaders import PageContent

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------- #
# Heuristic lookups
# --------------------------------------------------------------------- #
KNOWN_MAKES: Final[tuple[str, ...]] = (
    "toyota",
    "honda",
    "ford",
    "chevrolet",
    "chevy",
    "nissan",
    "bmw",
    "mercedes",
    "volkswagen",
    "vw",
    "audi",
    "subaru",
    "mazda",
    "hyundai",
    "kia",
    "dodge",
    "jeep",
    "chrysler",
    "gmc",
    "lexus",
    "acura",
    "porsche",
    "cadillac",
    "buick",
    "ram",
    "volvo",
    "mitsubishi",
    "mini",
    "tesla",
    "land rover",
    "jaguar",
    "infiniti",
    "genesis",
)

# Normalised make -> likely models. Lookup is case-insensitive, substrings match.
KNOWN_MODELS: Final[dict[str, tuple[str, ...]]] = {
    "toyota": ("camry", "corolla", "rav4", "highlander", "tacoma", "tundra", "sienna", "prius", "4runner", "yaris", "supra", "land cruiser"),
    "honda": ("civic", "accord", "cr-v", "crv", "pilot", "odyssey", "hr-v", "hrv", "fit", "passport", "ridgeline"),
    "ford": ("f-150", "mustang", "explorer", "escape", "focus", "fusion", "edge", "ranger", "expedition", "bronco", "fiesta", "transit"),
    "chevrolet": ("silverado", "malibu", "equinox", "tahoe", "suburban", "camaro", "impala", "cruze", "traverse", "colorado", "spark", "tahoe"),
    "nissan": ("altima", "sentra", "rogue", "frontier", "pathfinder", "versa", "armada", "maxima", "leaf"),
    "bmw": ("3 series", "5 series", "x3", "x5", "m3", "m5", "x1", "7 series"),
    "mercedes": ("c-class", "e-class", "s-class", "gla", "glc", "gle", "c300", "e300"),
    "volkswagen": ("golf", "jetta", "passat", "tiguan", "atlas", "beetle", "taos"),
    "subaru": ("outback", "forester", "impreza", "crosstrek", "legacy", "ascent", "brz"),
    "mazda": ("mazda3", "mazda6", "cx-5", "cx-30", "cx-9", "mx-5", "miata"),
    "hyundai": ("elantra", "sonata", "tucson", "santa fe", "kona", "palisade", "accent", "ioniq"),
    "kia": ("sportage", "sorento", "telluride", "forte", "k5", "soul", "niro", "rio"),
    "lexus": ("rx", "es", "nx", "gx", "lx", "is", "rc", "ux"),
    "acura": ("tlx", "rdx", "mdx", "ilx", "nsx", "integra"),
    "jeep": ("wrangler", "grand cherokee", "cherokee", "compass", "renegade", "gladiator"),
}

_DOC_TYPE_PATTERNS: Final[tuple[tuple[re.Pattern[str], DocType], ...]] = (
    (re.compile(r"\bdtc\b|trouble\s*code|diagnostic\s*trouble", re.IGNORECASE), DOCUMENT_TYPE_DTC),
    (re.compile(r"\btsb\b|technical\s*service\s*bulletin|bulletin", re.IGNORECASE), DOCUMENT_TYPE_TSB),
    (re.compile(r"wiring|electrical\s*(diagram|schematic)|diagram", re.IGNORECASE), DOCUMENT_TYPE_WIRING_DIAGRAM),
    (re.compile(r"\brepair\s*manual\b|shop\s*manual\b", re.IGNORECASE), DOCUMENT_TYPE_REPAIR_MANUAL),
)

_YEAR_PATTERN = re.compile(r"\b(19[6-9]\d|20[0-4]\d)\b")
_ENGINE_PATTERN = re.compile(r"\b\d(?:\.\d)?l(?:\b|$)", re.IGNORECASE)
_VIN_PATTERN = re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b")
_MODEL_YEAR_STRING = re.compile(r"(19[6-9]\d|20[0-4]\d)")

_SOURCE_HINTS = 3  # number of leading pages scanned for hints


@dataclass(frozen=True)
class DocumentMetadata:
    """Extracted document metadata (immutable)."""

    title: str
    doc_type: DocType
    make: str | None = None
    model: str | None = None
    year: int | None = None
    engine: str | None = None
    vin: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return metadata as a flat dict for vector-store storage."""
        return {
            "title": self.title,
            "doc_type": self.doc_type,
            "make": self.make or "",
            "model": self.model or "",
            "year": self.year or "",
            "engine": self.engine or "",
            "vin": self.vin or "",
        }


def _infer_doc_type(text: str) -> DocType:
    for pattern, doc_type in _DOC_TYPE_PATTERNS:
        if pattern.search(text):
            return doc_type
    return DOCUMENT_TYPE_SERVICE_MANUAL


def _find_make(text: str) -> str | None:
    lowered = text.lower()
    for make in KNOWN_MAKES:
        if re.search(rf"\b{re.escape(make)}\b", lowered):
            return make.title()
    return None


def _find_model(text: str, make: str | None) -> str | None:
    lowered = text.lower()
    if make:
        for model in KNOWN_MODELS.get(make.lower(), ()):
            if re.search(rf"\b{re.escape(model)}\b", lowered):
                return model.title()
    return None


def _find_year(text: str) -> int | None:
    years = [int(m) for m in _YEAR_PATTERN.findall(text)]
    if not years:
        return None
    return max(set(years), key=years.count)


def _find_engine(text: str) -> str | None:
    match = _ENGINE_PATTERN.search(text)
    return match.group(0).upper() if match else None


def _find_vin(text: str) -> str | None:
    match = _VIN_PATTERN.search(text)
    return match.group(0) if match else None


class MetadataExtractor:
    """Extract :class:`DocumentMetadata` from a file and its pages."""

    def __init__(self, source_hints: int = _SOURCE_HINTS) -> None:
        self.source_hints = source_hints

    def extract(
        self,
        path: Path,
        pages: list[PageContent],
        doc_type: DocType | None = None,
    ) -> DocumentMetadata:
        """Extract metadata combining filename hints and leading pages.

        Args:
            path: Source file path (filename contributes strong hints).
            pages: Loaded pages (leading pages scanned for content hints).
            doc_type: Explicit document type override.
        """
        filename = path.stem.replace("_", " ").replace("-", " ").replace(".", " ")
        hints = " ".join(page.text for page in pages[: self.source_hints])
        haystack = f"{filename}\n{hints}"

        title = _clean_title(path.stem)
        inferred_type = doc_type or _infer_doc_type(f"{filename} {hints}")
        make = _find_make(haystack)
        model = _find_model(haystack, make)
        year = _find_year(haystack)
        engine = _find_engine(haystack)
        vin = _find_vin(haystack)

        return DocumentMetadata(
            title=title,
            doc_type=inferred_type,
            make=make,
            model=model,
            year=year,
            engine=engine,
            vin=vin,
        )


def _clean_title(stem: str) -> str:
    """Turn a filename stem into a readable title."""
    title = re.sub(r"[_-]+", " ", stem).strip()
    return title if title else "Untitled"
