"""Unit tests for company name normalization."""

from __future__ import annotations

from services.alumni.normalizer import normalize_company_name


def test_suffix_stripping() -> None:
    cases = [
        ("Google Inc.", "google"),
        ("Spotify Ltd", "spotify"),
        ("Apple LLC", "apple"),
        ("TCS Technologies Pvt Ltd", "tcs"),
        ("Acme Corporation", "acme"),
        ("Stripe, Inc.", "stripe"),
        ("Oracle Technologies", "oracle"),
        ("Siemens AG", "siemens ag"),  # non-stripped specific abbreviation
        ("Zomato Private Limited", "zomato"),
    ]
    for raw, expected in cases:
        norm = normalize_company_name(raw)
        assert norm == expected, f"Failed for '{raw}': got '{norm}', expected '{expected}'"


def test_trailing_the_and_leading_the() -> None:
    assert normalize_company_name("The New York Times") == "new york times"
    assert normalize_company_name("Economist, The") == "economist"


def test_original_name_preserved() -> None:
    original = "Apex Technologies Pvt. Ltd."
    norm = normalize_company_name(original)
    assert original == "Apex Technologies Pvt. Ltd."
    assert norm == "apex"


def test_unicode_normalization() -> None:
    # NFKC normalizes fullwidth characters and ligature
    raw = "Ｇｏｏｇｌｅ Ｉｎｃ．"
    norm = normalize_company_name(raw)
    assert norm == "google"
