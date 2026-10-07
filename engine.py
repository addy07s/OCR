"""Base64 document -> OpenCV image -> OCR text lines. Everything stays in memory."""

import base64
import binascii
import threading
from functools import lru_cache

import cv2
import fitz  # PyMuPDF
import numpy as np
from paddleocr import PaddleOCR

PDF_RENDER_ZOOM: float = 2.0  # 2x of 72 DPI, a good balance of OCR accuracy and speed
MIN_CONFIDENCE: float = 0.5  # text PaddleOCR is less sure about is dropped

_inference_lock = threading.Lock()  # Paddle inference is not guaranteed thread-safe


class DocumentDecodeError(Exception):
    """The Base64 string or the document inside it is invalid."""


@lru_cache(maxsize=1)
def get_ocr_engine() -> PaddleOCR:
    """Create the PaddleOCR model once and reuse it for every request."""
    return PaddleOCR(lang="en", use_angle_cls=True, show_log=False)


def _decode_base64(base64_string: str) -> bytes:
    try:
        data = base64.b64decode(base64_string)
    except binascii.Error as exc:
        raise DocumentDecodeError("document_base64 is not valid Base64.") from exc
    if not data:
        raise DocumentDecodeError("document_base64 is empty.")
    return data


def _render_pdf(pdf_bytes: bytes) -> np.ndarray:
    """Render the first page of a PDF as a BGR image."""
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
            matrix = fitz.Matrix(PDF_RENDER_ZOOM, PDF_RENDER_ZOOM)
            pixmap = pdf[0].get_pixmap(matrix=matrix, colorspace=fitz.csRGB, alpha=False)
            rgb = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                pixmap.height, pixmap.width, pixmap.n
            )
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception as exc:
        raise DocumentDecodeError("Unable to read the PDF document.") from exc


def _decode_image(image_bytes: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise DocumentDecodeError("Unsupported or corrupt image data.")
    return image


def _read_text(image: np.ndarray) -> list[str]:
    """Run PaddleOCR on one image and return the recognised lines."""
    with _inference_lock:
        result = get_ocr_engine().ocr(image, cls=True)
    detections = result[0] or []  # None when no text was found
    return [text.strip() for _box, (text, confidence) in detections if confidence >= MIN_CONFIDENCE]


def run_ocr(base64_string: str) -> list[str]:
    """Decode a Base64 PDF or image and return its text lines in reading order.

    Raises:
        DocumentDecodeError: the Base64 string or the document is invalid.
    """
    data = _decode_base64(base64_string)
    image = _render_pdf(data) if data.startswith(b"%PDF-") else _decode_image(data)
    return _read_text(image)
