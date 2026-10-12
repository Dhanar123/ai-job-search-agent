"""Smoke tests for the resume parser and candidate profile services.

Run with:  python tests/test_step2.py
"""

from __future__ import annotations

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.resume_parser import extract_text_from_pdf, parse_resume, ResumeParseError


def _make_minimal_pdf(text: str = "Hello World") -> bytes:
    """Create a tiny valid PDF in memory using PyMuPDF."""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 72), text, fontsize=12)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_extract_valid_pdf() -> None:
    content = "John Doe — Software Engineer with 5 years of Python experience"
    pdf = _make_minimal_pdf(content)
    result = extract_text_from_pdf(pdf)
    assert "John Doe" in result, f"Expected 'John Doe' in result, got: {result!r}"
    print("  ✅ test_extract_valid_pdf")


def test_empty_bytes_raises() -> None:
    try:
        extract_text_from_pdf(b"")
        assert False, "Should have raised ResumeParseError"
    except ResumeParseError:
        pass
    print("  ✅ test_empty_bytes_raises")


def test_invalid_pdf_raises() -> None:
    try:
        extract_text_from_pdf(b"this is not a pdf")
        assert False, "Should have raised ResumeParseError"
    except ResumeParseError:
        pass
    print("  ✅ test_invalid_pdf_raises")


def test_unsupported_format_raises() -> None:
    try:
        parse_resume(b"data", "resume.docx")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("  ✅ test_unsupported_format_raises")


def test_parse_resume_pdf() -> None:
    content = "Jane Smith — Data Scientist with ML and deep learning skills"
    pdf = _make_minimal_pdf(content)
    result = parse_resume(pdf, "resume.pdf")
    assert "Jane Smith" in result
    print("  ✅ test_parse_resume_pdf")


def test_image_only_pdf_raises() -> None:
    """A PDF with no text content should be detected as unreadable."""
    import pymupdf

    doc = pymupdf.open()
    doc.new_page(width=612, height=792)  # blank page, no text
    pdf_bytes = doc.tobytes()
    doc.close()

    try:
        extract_text_from_pdf(pdf_bytes)
        assert False, "Should have raised ResumeParseError for empty PDF"
    except ResumeParseError:
        pass
    print("  ✅ test_image_only_pdf_raises")


def test_build_candidate_profile_groq() -> None:
    """Test candidate profile extraction using Groq API."""
    from dotenv import load_dotenv
    load_dotenv()

    from services.candidate_profile import build_candidate_profile, CandidateProfile

    sample_resume = (
        "Alice Johnson\n"
        "Email: alice.j@example.com | Phone: 123-456-7890\n"
        "Location: Seattle, WA\n"
        "Summary: Senior Backend Engineer with 6 years experience in Python and cloud systems.\n"
        "Skills: Python, FastAPI, Docker, PostgreSQL, AWS\n"
        "Experience:\n"
        "Senior Developer at CloudTech (2020 - Present) - Built scalable cloud APIs.\n"
        "Education:\n"
        "B.S. in Computer Science, University of Washington (2016 - 2020)\n"
    )

    profile = build_candidate_profile(sample_resume)
    assert isinstance(profile, CandidateProfile)
    assert "Alice" in profile.name
    assert "Python" in profile.skills
    assert len(profile.experience) >= 1
    print(f"  ✅ test_build_candidate_profile_groq (Extracted {profile.name}, {len(profile.skills)} skills)")


if __name__ == "__main__":
    print("Running Step 2 smoke tests...\n")
    test_extract_valid_pdf()
    test_empty_bytes_raises()
    test_invalid_pdf_raises()
    test_unsupported_format_raises()
    test_parse_resume_pdf()
    test_image_only_pdf_raises()
    test_build_candidate_profile_groq()
    print("\n🎉 All tests passed!")
