"""Deep-learning core: Base64 document -> OpenCV image -> OCR text lines.

One document (one image, or the first page of a PDF) is processed per call.
Everything happens in memory; no file is ever written to disk.
"""

import base64
import binascii
import logging
import re
import threading
from functools import lru_cache
from io import BytesIO
from typing import List

import cv2
import fitz  # PyMuPDF
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image

logger = logging.getLogger(__name__)

PDF_RENDER_ZOOM: float = 2.0  # 2x of 72 DPI -> ~144 DPI, a good OCR/speed balance
MIN_CONFIDENCE: float = 0.5  # discard text that PaddleOCR is unsure about

_DATA_URI_PREFIX = re.compile(r"^data:[^;,]+;base64,", re.IGNORECASE)
_inference_lock = threading.Lock()  # Paddle inference is not guaranteed thread-safe


class DocumentDecodeError(ValueError):
    """The Base64 string or the document inside it is invalid (client error)."""


class OcrProcessingError(RuntimeError):
    """The OCR model failed while processing a valid document (server error)."""


@lru_cache(maxsize=1)
def get_ocr_engine() -> PaddleOCR:
    """Create the PaddleOCR model once and reuse it for every request."""
    logger.info("Loading PaddleOCR model...")
    return PaddleOCR(lang="en", use_angle_cls=True, show_log=False)


def _decode_base64(base64_string: str) -> bytes:
    """Decode a Base64 string (optionally a data URI) into raw bytes."""
    cleaned = _DATA_URI_PREFIX.sub("", base64_string.strip())
    cleaned = "".join(cleaned.split())  # drop any line breaks / spaces
    try:
        data = base64.b64decode(cleaned, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise DocumentDecodeError("document_base64 is not valid Base64.") from exc
    if not data:
        raise DocumentDecodeError("document_base64 is empty.")
    return data


def _pdf_to_image(pdf_bytes: bytes) -> np.ndarray:
    """Render the first PDF page to a BGR OpenCV image."""
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
            matrix = fitz.Matrix(PDF_RENDER_ZOOM, PDF_RENDER_ZOOM)
            pix = pdf[0].get_pixmap(matrix=matrix, colorspace=fitz.csRGB, alpha=False)
            rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception as exc:
        raise DocumentDecodeError("Unable to read the PDF document.") from exc


def _decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode an image with OpenCV, falling back to Pillow for exotic formats."""
    decoded = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is not None:
        return decoded
    try:
        with Image.open(BytesIO(image_bytes)) as pil_image:
            rgb = np.array(pil_image.convert("RGB"))
    except Exception as exc:
        raise DocumentDecodeError("Unsupported or corrupt image/PDF data.") from exc
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _bytes_to_image(data: bytes) -> np.ndarray:
    """Route the raw bytes to the PDF or image loader based on the file header."""
    if b"%PDF-" in data[:1024]:
        return _pdf_to_image(data)
    return _decode_image(data)


def _extract_text(image: np.ndarray) -> List[str]:
    """Run PaddleOCR on the image and return a flat list of text lines."""
    engine = get_ocr_engine()
    try:
        with _inference_lock:
            result = engine.ocr(image, cls=True)
    except Exception as exc:
        logger.exception("PaddleOCR inference failed")
        raise OcrProcessingError("OCR inference failed.") from exc

    # PaddleOCR 2.x wraps the single image's lines in a list; it is None if no text was found.
    detections = (result or [None])[0] or []
    return [
        text.strip()
        for _box, (text, confidence) in detections
        if confidence >= MIN_CONFIDENCE and text.strip()
    ]


def run_ocr(base64_string: str) -> List[str]:
    """Decode a Base64 PDF/image and return all recognised text lines in reading order.

    Raises:
        DocumentDecodeError: invalid Base64 or unreadable document.
        OcrProcessingError: the OCR model failed.
    """
    data = _decode_base64(base64_string)
    image = _bytes_to_image(data)
    return _extract_text(image)
