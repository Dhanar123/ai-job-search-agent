"""Resume parser service.

Extracts text from uploaded PDF resumes using PyMuPDF.
Handles corrupted files, empty documents, and scanned/image-only PDFs.
Uses multiple extraction strategies for maximum compatibility.
"""

from __future__ import annotations

import logging

import pymupdf  # PyMuPDF

logger = logging.getLogger(__name__)

# Minimum character count to consider a PDF "readable"
_MIN_TEXT_LENGTH = 30


class ResumeParseError(Exception):
    """Raised when a resume cannot be parsed."""


def _extract_with_text_mode(doc: pymupdf.Document) -> str:
    """Primary extraction: plain text mode."""
    pages: list[str] = []
    for page in doc:
        text = page.get_text("text")
        if text and text.strip():
            pages.append(text.strip())
    return "\n\n".join(pages)


def _extract_with_blocks_mode(doc: pymupdf.Document) -> str:
    """Fallback extraction: block-level text extraction.

    More resilient to unusual font encodings and text-as-curves.
    """
    pages: list[str] = []
    for page in doc:
        blocks = page.get_text("blocks")
        # Each block is a tuple: (x0, y0, x1, y1, text, block_no, block_type)
        # block_type 0 = text
        page_lines: list[str] = []
        for block in blocks:
            if block[6] == 0:  # text block
                line = block[4].strip()
                if line:
                    page_lines.append(line)
        if page_lines:
            pages.append("\n".join(page_lines))
    return "\n\n".join(pages)


def _extract_with_dict_mode(doc: pymupdf.Document) -> str:
    """Last-resort extraction: dict mode parses individual spans.

    Handles PDFs where text is embedded in complex structures.
    """
    pages: list[str] = []
    for page in doc:
        data = page.get_text("dict")
        page_lines: list[str] = []
        for block in data.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans_text = "".join(
                    span.get("text", "") for span in line.get("spans", [])
                )
                if spans_text.strip():
                    page_lines.append(spans_text.strip())
        if page_lines:
            pages.append("\n".join(page_lines))
    return "\n\n".join(pages)


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from every page of a PDF byte stream.

    Tries three extraction strategies in order of efficiency:
    1. Plain text mode (fastest, works for most PDFs)
    2. Block-level extraction (handles unusual encodings)
    3. Dict/span-level extraction (handles complex structures)

    Args:
        pdf_bytes: Raw bytes of a PDF file.

    Returns:
        Concatenated text from all pages.

    Raises:
        ResumeParseError: If the file is not a valid PDF, is empty,
            or no strategy could extract sufficient text.
    """
    if not pdf_bytes:
        raise ResumeParseError("Received empty file — no bytes to process.")

    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ResumeParseError(f"Unable to open PDF: {exc}") from exc

    if doc.page_count == 0:
        doc.close()
        raise ResumeParseError("PDF has zero pages.")

    # Try each extraction strategy, keep the best result
    strategies = [
        ("text", _extract_with_text_mode),
        ("blocks", _extract_with_blocks_mode),
        ("dict", _extract_with_dict_mode),
    ]

    best_text = ""
    for name, strategy in strategies:
        try:
            result = strategy(doc)
            logger.debug("Strategy '%s' extracted %d chars.", name, len(result))
            if len(result) > len(best_text):
                best_text = result
            # If we got a good result, stop trying
            if len(best_text) >= _MIN_TEXT_LENGTH:
                break
        except Exception as exc:
            logger.debug("Strategy '%s' failed: %s", name, exc)
            continue

    doc.close()

    if len(best_text) < _MIN_TEXT_LENGTH:
        raise ResumeParseError(
            "PDF appears to be empty or image-only — "
            f"extracted only {len(best_text)} characters. "
            "Please upload a text-based PDF."
        )

    logger.info("Extracted %d characters from PDF.", len(best_text))
    return best_text


def parse_resume(file_bytes: bytes, filename: str) -> str:
    """Parse a resume file and return the extracted text.

    Currently supports PDF only.

    Args:
        file_bytes: Raw bytes of the uploaded file.
        filename: Original filename (used to detect format).

    Returns:
        Extracted resume text.

    Raises:
        ValueError: If the file format is unsupported.
        ResumeParseError: If text extraction fails.
    """
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext == "pdf":
        return extract_text_from_pdf(file_bytes)
    else:
        raise ValueError(
            f"Unsupported file format: .{ext}  — only PDF is supported."
        )
