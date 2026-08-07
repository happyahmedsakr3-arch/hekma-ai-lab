# Compatibility shim: V5 now uses PaddleOCR as the primary OCR engine.
# Existing imports keep working without touching the Streamlit UI code.
from src.paddle_engine import (
    extract_insurance_text,
    extract_national_id,
    national_id_roi,
    normalize_digits,
)

__all__ = [
    "extract_insurance_text",
    "extract_national_id",
    "national_id_roi",
    "normalize_digits",
]
