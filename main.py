"""API gateway: receives Base64 documents from Salesforce and returns parsed fields."""

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

import engine
import parser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ocr-api")


MAX_BASE64_CHARS: int = 8_000_000  # ~6 MB of decoded file data


class OcrPayload(BaseModel):
    """Request body sent by Salesforce (exactly one document per request)."""

    document_base64: str = Field(..., min_length=1, max_length=MAX_BASE64_CHARS)
    document_type: Literal["AADHAAR", "PAN"]

    @field_validator("document_type", mode="before")
    @classmethod
    def normalise_document_type(cls, value: Any) -> Any:
        """Accept 'pan', ' Aadhaar ' etc. by upper-casing and trimming."""
        return value.strip().upper() if isinstance(value, str) else value


class OcrResponse(BaseModel):
    """Response body returned to Salesforce."""

    success: bool
    data: Dict[str, Any]
    execution_time: float


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Load the OCR model at startup so the first request is not slow."""
    engine.get_ocr_engine()
    yield


app = FastAPI(title="OCR Microservice", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> Dict[str, str]:
    """Liveness probe for load balancers."""
    return {"status": "ok"}


# Plain `def` (not `async def`) so FastAPI runs this CPU-heavy work in a worker
# thread instead of blocking the event loop.
@app.post("/api/ocr", response_model=OcrResponse)
def ocr_endpoint(payload: OcrPayload) -> OcrResponse:
    """Run OCR on a Base64 document and return the structured fields."""
    start = time.perf_counter()
    try:
        raw_text = engine.run_ocr(payload.document_base64)
        data = parser.parse_document_data(raw_text, payload.document_type)
    except (engine.DocumentDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except engine.OcrProcessingError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        # Never leak internals (or document contents) to the caller.
        logger.exception("Unexpected failure while processing document")
        raise HTTPException(status_code=500, detail="Internal server error.") from exc

    return OcrResponse(
        success=True,
        data=data,
        execution_time=round(time.perf_counter() - start, 3),
    )
