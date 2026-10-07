"""API gateway: accepts a Base64 document and returns the extracted fields."""

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import engine
import parser

MAX_BASE64_CHARS: int = 8_000_000  # about 6 MB of file data

logger = logging.getLogger(__name__)


class OcrPayload(BaseModel):
    document_base64: str = Field(min_length=1, max_length=MAX_BASE64_CHARS)
    document_type: Literal["AADHAAR", "PAN"]


class OcrResponse(BaseModel):
    success: bool
    data: dict[str, Optional[str]]
    execution_time: float


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Load the OCR model at startup so the first request is not slow."""
    engine.get_ocr_engine()
    yield


app = FastAPI(title="OCR Microservice", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# A plain `def` makes FastAPI run this CPU-heavy work in a worker thread,
# so it does not block the event loop.
@app.post("/api/ocr")
def ocr(payload: OcrPayload) -> OcrResponse:
    start = time.perf_counter()
    try:
        text_lines = engine.run_ocr(payload.document_base64)
        data = parser.parse_document_data(text_lines, payload.document_type)
    except engine.DocumentDecodeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("OCR processing failed")
        raise HTTPException(status_code=500, detail="OCR processing failed.") from exc

    return OcrResponse(
        success=True,
        data=data,
        execution_time=round(time.perf_counter() - start, 3),
    )
