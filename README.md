# OCR Microservice (Aadhaar / PAN)

A small web service that reads an Aadhaar or PAN document and returns the details on it as JSON.

## How the OCR works

1. **Receive:** the client sends a JSON request with the document as a Base64 string and a document type (`AADHAAR` or `PAN`).
2. **Convert:** the Base64 text is decoded in memory (nothing is saved to disk). A PDF is rendered to an image with PyMuPDF (first page only). An image file is loaded with OpenCV.
3. **Read:** PaddleOCR (English, with text-angle correction) finds the text in the image. Text it is not confident about is dropped.
4. **Extract:** regular expressions pick the fields out of the text.
   - **Aadhaar:** name, date of birth, gender and the 12-digit Aadhaar number.
   - **PAN:** PAN number (5 letters, 4 digits, 1 letter), name, father's name and date of birth.
5. **Respond:** the fields come back as JSON. A field that could not be found is `null`.

Only one document is processed per request. The Base64 string is limited to 8,000,000 characters (about 6 MB of file).

## What to install first

| Requirement | Details |
|---|---|
| **Python 3.9 - 3.12** | Download from [python.org](https://www.python.org/downloads/). On Windows, tick **"Add Python to PATH"** during install. Python 3.13 is not supported by the pinned PaddlePaddle version. |
| **Microsoft Visual C++ Redistributable** (Windows only) | Needed by PaddlePaddle. Get it from [Microsoft](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist). |
| **Internet connection** | Needed for `pip install`, and PaddleOCR downloads its models the first time the app starts. |
| **About 2 GB free disk space and 4 GB RAM** | Recommended. |

All Python libraries (FastAPI, PaddleOCR, PyMuPDF, OpenCV and so on) are installed by `pip` in the next step.

Check that Python works:

```bash
python --version
```

## Install

Open a terminal in the project folder and run:

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

If PowerShell blocks the activate script, run this once and try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The first start takes a minute or two while the OCR models download. The service is ready when you see `Uvicorn running on http://0.0.0.0:8000`.

- Health check: <http://localhost:8000/health>
- Interactive docs where you can try the API: <http://localhost:8000/docs>

## Use the API

`POST /api/ocr`

```json
{
  "document_base64": "<Base64 text of a PDF, JPG or PNG>",
  "document_type": "PAN"
}
```

The response looks like this. The fields depend on the document type.

```json
{
  "success": true,
  "data": {
    "pan_number": "<value or null>",
    "name": "<value or null>",
    "father_name": "<value or null>",
    "dob": "<value or null>"
  },
  "execution_time": 1.482
}
```

| Status | Meaning |
|---|---|
| `400` | The Base64 is invalid, the file is not a readable PDF or image, or the document type is unknown. |
| `422` | The request body is wrong: a missing field, a bad `document_type`, or a payload over the size limit. |
| `500` | The OCR failed or something unexpected happened on the server. |

Quick test from Windows PowerShell:

```powershell
$b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes("C:\path\to\document.jpg"))
$body = @{ document_base64 = $b64; document_type = "PAN" } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/api/ocr -Method Post -ContentType "application/json" -Body $body
```

## Project files

| File | What it does |
|---|---|
| `main.py` | The web API. Receives the request and returns the response. |
| `engine.py` | Decodes the Base64 document, turns it into an image and runs OCR. |
| `parser.py` | Uses regex to pull the fields out of the OCR text. |
| `requirements.txt` | The Python libraries to install. |

## Troubleshooting

| Problem | Fix |
|---|---|
| `pip install` fails on `paddlepaddle` | Check `python --version`. It must be 3.9 to 3.12 and 64-bit. |
| `ImportError: DLL load failed` (Windows) | Install the Microsoft Visual C++ Redistributable listed above. |
| First start is slow | The models are downloading. Check your internet connection and wait. |
| Fields come back as `null` | The photo is probably blurry, dark or cropped. Use a clear, flat, well-lit scan. |
