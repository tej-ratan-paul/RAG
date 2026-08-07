"""Tests for metadata extraction."""

from __future__ import annotations

from pathlib import Path

from auto_rag.constants import (
    DOCUMENT_TYPE_DTC,
    DOCUMENT_TYPE_REPAIR_MANUAL,
    DOCUMENT_TYPE_SERVICE_MANUAL,
    DOCUMENT_TYPE_TSB,
    DOCUMENT_TYPE_WIRING_DIAGRAM,
)
from auto_rag.ingestion.loaders import PageContent
from auto_rag.ingestion.metadata import MetadataExtractor

extractor = MetadataExtractor(source_hints=3)


def test_doc_type_from_filename() -> None:
    pages = [PageContent(page_number=1, text="Contents of a manual.")]
    assert extractor.extract(Path("toyota_camry_2018_manual.pdf"), pages).doc_type == DOCUMENT_TYPE_SERVICE_MANUAL
    assert extractor.extract(Path("dtc_p0301_codes.pdf"), pages).doc_type == DOCUMENT_TYPE_DTC
    assert extractor.extract(Path("tsb_0147_engine.pdf"), pages).doc_type == DOCUMENT_TYPE_TSB
    assert extractor.extract(Path("camry_wiring_diagram.pdf"), pages).doc_type == DOCUMENT_TYPE_WIRING_DIAGRAM
    assert extractor.extract(Path("corolla_repair_manual.pdf"), pages).doc_type == DOCUMENT_TYPE_REPAIR_MANUAL


def test_make_model_year_extraction() -> None:
    pages = [PageContent(page_number=1, text="2018 Toyota Camry 2.5L engine service information.")]
    meta = extractor.extract(Path("service_information.pdf"), pages)
    assert meta.make == "Toyota"
    assert meta.model == "Camry"
    assert meta.year == 2018
    assert meta.engine == "2.5L"


def test_vin_extraction() -> None:
    pages = [PageContent(page_number=1, text="VIN 4T1B11HK3JU123456 shown on the door label.")]
    meta = extractor.extract(Path("camry.pdf"), pages)
    assert meta.vin == "4T1B11HK3JU123456"


def test_doc_type_override_wins() -> None:
    pages = [PageContent(page_number=1, text="some manual content")]
    meta = extractor.extract(Path("mystery.pdf"), pages, doc_type=DOCUMENT_TYPE_TSB)
    assert meta.doc_type == DOCUMENT_TYPE_TSB


def test_title_uses_clean_filename() -> None:
    pages = [PageContent(page_number=1, text="x")]
    meta = extractor.extract(Path("2018_Toyota_Camry_Engine.pdf"), pages)
    assert meta.title == "2018 Toyota Camry Engine"


def test_no_metadata_found() -> None:
    pages = [PageContent(page_number=1, text="ordinary content with no hints whatsoever")]
    meta = extractor.extract(Path("doc123.pdf"), pages)
    assert meta.make is None
    assert meta.model is None
    assert meta.year is None
    assert meta.doc_type == DOCUMENT_TYPE_SERVICE_MANUAL
