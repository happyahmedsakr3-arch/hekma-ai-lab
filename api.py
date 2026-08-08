import os

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.insurance_service import analyze_insurance_card, collector_payload

app = FastAPI(title="Hekma AI Insurance Reader API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


def _api_key() -> str:
    return os.getenv("OPENAI_API_KEY", "")


def _check_reader_key(x_reader_key: str | None):
    expected = os.getenv("HEKMA_READER_KEY", "").strip()
    if expected and (x_reader_key or "").strip() != expected:
        raise HTTPException(status_code=401, detail="Invalid reader key")


@app.get("/health")
def health():
    return {"ok": True, "service": "hekma-ai-insurance-reader"}


@app.post("/api/read-insurance-card")
async def read_insurance_card(
    file: UploadFile = File(...),
    x_reader_key: str | None = Header(default=None),
):
    _check_reader_key(x_reader_key)
    key = _api_key()
    if not key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty image")

    if len(raw) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image is too large")

    try:
        bundle = analyze_insurance_card(raw, api_key=key, model=os.getenv("OPENAI_MODEL", "gpt-5.1"))
        return JSONResponse(collector_payload(bundle))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Insurance card analysis failed: {exc}")
