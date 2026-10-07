"""Turns OCR text lines into structured fields using regular expressions."""

import re
from typing import Callable, Optional

Fields = dict[str, Optional[str]]

_FLAGS = re.IGNORECASE | re.MULTILINE

_DATE = r"\d{2}[/-]\d{2}[/-]\d{4}"  # DD/MM/YYYY or DD-MM-YYYY
_DOB_LABEL = r"(?:DOB|D\.O\.B|Date\s+of\s+Birth|Year\s+of\s+Birth|YOB)"
_NAME = r"[A-Za-z][A-Za-z .'\-]+"  # letters, spaces, dots, apostrophes, hyphens

# A date (or a bare year) right after a DOB label, e.g. "DOB: 01/01/1990" or "Year of Birth: 1990".
# \s* lets the value sit on the next line.
_LABELLED_DOB = re.compile(rf"{_DOB_LABEL}\s*[:\-]?\s*({_DATE}|\d{{4}})", re.IGNORECASE)
_ANY_DATE = re.compile(rf"({_DATE})")

# 12 digits starting with 2-9, optionally grouped 4-4-4. The lookarounds reject digits
# that belong to a longer number, so the 16-digit VID is not read as an Aadhaar number.
_AADHAAR_NUMBER = re.compile(r"(?<!\d)(?<!\d )([2-9]\d{3} ?\d{4} ?\d{4})(?! ?\d)")

# On the card the name is the line directly above the DOB line.
_AADHAAR_NAME = re.compile(rf"^({_NAME})\n(?:{_DOB_LABEL}|{_DATE})", _FLAGS)

# FEMALE is listed before MALE so the longer word wins.
_GENDER = re.compile(r"\b(FEMALE|MALE|TRANSGENDER)\b", re.IGNORECASE)

# 5 letters + 4 digits + 1 letter, e.g. ABCDE1234F.
_PAN_NUMBER = re.compile(r"\b([A-Z]{5}\d{4}[A-Z])\b")

# A "Name" label on a line without "father"; the value follows on the same or the next line.
# The lookahead stops a missing value from capturing the next label.
_PAN_NAME = re.compile(
    rf"^(?!.*father).*\bname\b\s*[:\-]?\s*(?!father|date)({_NAME})$", _FLAGS
)

# "Father's Name" label (apostrophe optional, OCR often drops it) and its value.
_PAN_FATHER_NAME = re.compile(
    rf"father['’`]?s?\s*name\s*[:\-]?\s*(?!date)({_NAME})$", _FLAGS
)


def _find(pattern: re.Pattern[str], text: str) -> Optional[str]:
    """Return the first capture group of the first match, or None."""
    match = pattern.search(text)
    return match.group(1).strip() if match else None


def _find_dob(text: str) -> Optional[str]:
    """Prefer a date next to a DOB label; otherwise fall back to the first date."""
    return _find(_LABELLED_DOB, text) or _find(_ANY_DATE, text)


def _parse_aadhaar(text: str) -> Fields:
    gender = _find(_GENDER, text)
    return {
        "name": _find(_AADHAAR_NAME, text),
        "dob": _find_dob(text),
        "gender": gender.capitalize() if gender else None,
        "aadhaar_number": _find(_AADHAAR_NUMBER, text),
    }


def _parse_pan(text: str) -> Fields:
    return {
        "pan_number": _find(_PAN_NUMBER, text),
        "name": _find(_PAN_NAME, text),
        "father_name": _find(_PAN_FATHER_NAME, text),
        "dob": _find_dob(text),
    }


_PARSERS: dict[str, Callable[[str], Fields]] = {
    "AADHAAR": _parse_aadhaar,
    "PAN": _parse_pan,
}


def parse_document_data(raw_text_list: list[str], document_type: str) -> Fields:
    """Extract the fields for `document_type` ("AADHAAR" or "PAN") from OCR text lines.

    Raises:
        ValueError: if the document type is not supported.
    """
    parse = _PARSERS.get(document_type)
    if parse is None:
        raise ValueError(f"Unsupported document_type: {document_type!r}")
    return parse("\n".join(raw_text_list))
