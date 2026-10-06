"""Turns raw OCR text lines into structured, PII-safe dictionaries."""

import re
from typing import Callable, Dict, List, Optional

Fields = Dict[str, Optional[str]]

AADHAAR_REDACTED = "[Aadhaar Redacted]"

# --- Shared patterns -------------------------------------------------------

# DD/MM/YYYY or DD-MM-YYYY.
_DATE = re.compile(r"(?<!\d)(\d{2}[/-]\d{2}[/-]\d{4})(?!\d)")

# A name line: letters, spaces, dots, apostrophes and hyphens only (no digits).
_NAME_LINE = re.compile(r"^[A-Za-z][A-Za-z .'\-]{2,}$")

# --- Aadhaar patterns ------------------------------------------------------

# 12 digits, first digit 2-9, optionally split into 4-4-4 groups. The lookarounds
# stop us from matching part of a longer number (e.g. the 16-digit VID).
_AADHAAR_NUMBER = re.compile(r"(?<!\d)(?<!\d )([2-9]\d{3})\s?(\d{4})\s?(\d{4})(?![ ]?\d)")

# "DOB: 01/01/1990", "Date of Birth 01-01-1990" or "Year of Birth: 1990"
# (the value may sit on the next line, hence \s*).
_AADHAAR_DOB = re.compile(
    r"(?:DOB|D\.O\.B|Date\s+of\s+Birth|Year\s+of\s+Birth|YOB)\s*[:\-]?\s*"
    r"(\d{2}[/-]\d{2}[/-]\d{4}|\d{4})",
    re.IGNORECASE,
)

# FEMALE is listed before MALE so the longer word wins.
_AADHAAR_GENDER = re.compile(r"\b(FEMALE|MALE|TRANSGENDER)\b", re.IGNORECASE)

# The line that holds (or introduces) the DOB; the name sits just above it.
_AADHAAR_DOB_ANCHOR = re.compile(
    r"DOB|D\.O\.B|Date\s+of\s+Birth|Year\s+of\s+Birth|YOB|\d{2}[/-]\d{2}[/-]\d{4}",
    re.IGNORECASE,
)

# Header/footer words that must never be mistaken for a person's name.
_AADHAAR_NOISE = re.compile(
    r"government|india|authority|unique|identification|aadhaar|aadhar|"
    r"birth|dob|male|female|address|vid|download|issue|help|s/o|d/o|w/o|c/o",
    re.IGNORECASE,
)

# --- PAN patterns ----------------------------------------------------------

# 5 letters + 4 digits + 1 letter, e.g. ABCDE1234F.
_PAN_NUMBER = re.compile(r"\b([A-Z]{5}\d{4}[A-Z])\b")

# "Father's Name" label (apostrophe optional because OCR often drops it).
_PAN_FATHER_LABEL = re.compile(r"father[’'`]?s?\s*name\s*[:\-]?\s*(.*)$", re.IGNORECASE)

# Plain "Name" label, used only on lines that are NOT the father label.
_PAN_NAME_LABEL = re.compile(r"\bname\s*[:\-]?\s*(.*)$", re.IGNORECASE)

# "Date of Birth" label, with the value on the same or the next line.
_PAN_DOB = re.compile(
    r"Date\s+of\s+Birth\s*[:\-]?\s*(\d{2}[/-]\d{2}[/-]\d{4})", re.IGNORECASE
)

# Printed card labels/headers that must not be returned as a person's name.
_PAN_NOISE = re.compile(
    r"income|tax|department|govt|government|india|permanent|account|number|"
    r"card|signature|date|birth|name",
    re.IGNORECASE,
)


# --- Helpers ---------------------------------------------------------------

def _clean(value: str) -> str:
    """Collapse whitespace and strip stray punctuation around a value."""
    return re.sub(r"\s+", " ", value).strip(" :-.,")


def _value_after_label(
    lines: List[str],
    label: re.Pattern,
    skip: Optional[re.Pattern] = None,
) -> Optional[str]:
    """Return the text following a label: on the same line, else on the next line."""
    for index, line in enumerate(lines):
        if skip and skip.search(line):
            continue
        match = label.search(line)
        if not match:
            continue
        inline = _clean(match.group(1)) if match.groups() else ""
        if _NAME_LINE.match(inline) and not _PAN_NOISE.search(inline):
            return inline
        if index + 1 < len(lines):
            following = _clean(lines[index + 1])
            if _NAME_LINE.match(following) and not _PAN_NOISE.search(following):
                return following
    return None


# --- Aadhaar ---------------------------------------------------------------

def _aadhaar_name(lines: List[str]) -> Optional[str]:
    """Name is the nearest clean alphabetic line above the DOB line."""
    for index, line in enumerate(lines):
        if _AADHAAR_DOB_ANCHOR.search(line):
            for candidate in reversed(lines[max(0, index - 3):index]):
                candidate = _clean(candidate)
                if _NAME_LINE.match(candidate) and not _AADHAAR_NOISE.search(candidate):
                    return candidate
            break
    return None


def _parse_aadhaar(lines: List[str]) -> Fields:
    """Extract Name, DOB, Gender and ID number; the ID number is always redacted."""
    text = "\n".join(lines)

    dob_match = _AADHAAR_DOB.search(text) or _DATE.search(text)
    gender_match = _AADHAAR_GENDER.search(text)
    number_found = _AADHAAR_NUMBER.search(" ".join(lines)) is not None

    return {
        "name": _aadhaar_name(lines),
        "dob": dob_match.group(1) if dob_match else None,
        "gender": gender_match.group(1).capitalize() if gender_match else None,
        # SECURITY: the real number is never stored in the output.
        "aadhaar_number": AADHAAR_REDACTED if number_found else None,
    }


# --- PAN -------------------------------------------------------------------

def _parse_pan(lines: List[str]) -> Fields:
    """Extract PAN number, Name, Father's Name and DOB."""
    text = "\n".join(lines)

    pan_match = _PAN_NUMBER.search(text.upper())
    dob_match = _PAN_DOB.search(text) or _DATE.search(text)

    return {
        "pan_number": pan_match.group(1) if pan_match else None,
        "name": _value_after_label(lines, _PAN_NAME_LABEL, skip=_PAN_FATHER_LABEL),
        "father_name": _value_after_label(lines, _PAN_FATHER_LABEL),
        "dob": dob_match.group(1) if dob_match else None,
    }


# --- Router ----------------------------------------------------------------

_PARSERS: Dict[str, Callable[[List[str]], Fields]] = {
    "AADHAAR": _parse_aadhaar,
    "PAN": _parse_pan,
}


def parse_document_data(raw_text_list: List[str], document_type: str) -> Fields:
    """Route OCR text to the right parser and return a clean dictionary.

    Raises:
        ValueError: if document_type is not "AADHAAR" or "PAN".
    """
    parser = _PARSERS.get(document_type.strip().upper())
    if parser is None:
        raise ValueError(f"Unsupported document_type: {document_type!r}")
    lines = [line.strip() for line in raw_text_list if line and line.strip()]
    return parser(lines)
