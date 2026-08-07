"""Tests for retrieval metadata filtering."""

from __future__ import annotations

import pytest

from auto_rag.retrieval.filters import matches_document, matches_filter, to_where
from auto_rag.retrieval.models import RetrievalFilter


def test_to_where_empty_filter() -> None:
    assert to_where(RetrievalFilter()) == {}


def test_to_where_single_condition() -> None:
    assert to_where(RetrievalFilter(make="Toyota")) == {"make": {"$eq": "Toyota"}}


def test_to_where_multiple_conditions_anded() -> None:
    where = to_where(RetrievalFilter(make="Toyota", doc_type="dtc"))
    assert where == {"$and": [{"make": {"$eq": "Toyota"}}, {"doc_type": {"$eq": "dtc"}}]}


def test_to_where_year_uses_plain_value() -> None:
    assert to_where(RetrievalFilter(year=2018)) == {"year": 2018}


def test_matches_filter_ignores_empty_string_metadata() -> None:
    filt = RetrievalFilter(make="Toyota")
    assert matches_filter(filt, {"make": "Toyota", "model": ""})
    assert not matches_filter(filt, {"make": "Honda"})


def test_matches_filter_case_insensitive() -> None:
    assert matches_filter(RetrievalFilter(make="toyota"), {"make": "Toyota"})


def test_matches_filter_year() -> None:
    assert matches_filter(RetrievalFilter(year=2018), {"year": 2018})
    assert not matches_filter(RetrievalFilter(year=2018), {"year": 2016})


def test_matches_filter_inactive_passes_any_metadata() -> None:
    assert matches_filter(RetrievalFilter(), {"make": "Anything"})


def test_filter_active_property() -> None:
    assert not RetrievalFilter().active
    assert RetrievalFilter(vin="4T1B11HK3JU123456").active


def test_matches_document_operators() -> None:
    assert matches_document({"make": "Toyota"}, {"make": "Toyota"})
    assert matches_document({"make": {"$ne": "Honda"}}, {"make": "Toyota"})
    assert matches_document({"year": {"$in": [2016, 2018]}}, {"year": 2018})
    assert matches_document({"year": {"$gte": 2017}}, {"year": 2018})
    assert not matches_document({"year": {"$gte": 2019}}, {"year": 2018})


def test_from_doc_type_validates() -> None:
    assert RetrievalFilter.from_doc_type("dtc").doc_type == "dtc"
    with pytest.raises(ValueError):
        RetrievalFilter.from_doc_type("not_a_type")
