import re
from typing import Any

import cv2
import numpy as np
import pytesseract
from PIL import Image

ARABIC_TO_LATIN = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def normalize_digits(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\D", "", text.translate(ARABIC_TO_LATIN))


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    if len(image.shape) == 2:
        return Image.fromarray(image)
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def deskew_and_enhance(image: Image.Image) -> Image.Image:
    frame = pil_to_bgr(image)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 35, 35)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    sharp = cv2.addWeighted(gray, 1.6, cv2.GaussianBlur(gray, (0, 0), 1.2), -0.6, 0)
    return bgr_to_pil(sharp)


def national_id_roi(image: Image.Image) -> Image.Image:
    frame = pil_to_bgr(image)
    h, w = frame.shape[:2]
    x1, y1 = int(w * 0.30), int(h * 0.58)
    x2, y2 = int(w * 0.995), int(h * 0.96)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return image
    target_w = max(2200, roi.shape[1] * 5)
    scale = target_w / roi.shape[1]
    roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 25, 25)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.addWeighted(gray, 1.8, cv2.GaussianBlur(gray, (0, 0), 1.0), -0.8, 0)
    return bgr_to_pil(gray)


def _ocr_text(image: Image.Image, lang: str, config: str = "--psm 6") -> str:
    return pytesseract.image_to_string(image, lang=lang, config=config)


def _national_id_structure_ok(value: str) -> bool:
    if len(value) != 14 or value[0] not in "23":
        return False
    try:
        import datetime as _dt
        century = 1900 if value[0] == "2" else 2000
        year = century + int(value[1:3])
        month = int(value[3:5])
        day = int(value[5:7])
        _dt.date(year, month, day)
    except Exception:
        return False
    return True


def _extract_candidates(text: str) -> list[str]:
    normalized = text.translate(ARABIC_TO_LATIN)
    groups = re.findall(r"[0-9][0-9\s.\-]{10,32}", normalized)
    candidates: list[str] = []
    for group in groups:
        digits = re.sub(r"\D", "", group)
        if len(digits) == 14:
            candidates.append(digits)
        elif len(digits) > 14:
            for i in range(len(digits) - 13):
                candidates.append(digits[i:i+14])
    return candidates


def extract_national_id(image: Image.Image) -> dict:
    roi = national_id_roi(image)
    cv_roi = np.array(roi.convert("L"))
    threshold = cv2.adaptiveThreshold(cv_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)
    threshold_img = Image.fromarray(threshold)

    variants = [
        ("roi_ara", roi, "ara", "--psm 6"),
        ("roi_eng", roi, "eng", "--psm 6"),
        ("threshold_ara", threshold_img, "ara", "--psm 6"),
        ("threshold_eng", threshold_img, "eng", "--psm 6"),
        ("digits", roi, "eng", "--psm 7 -c tessedit_char_whitelist=0123456789"),
    ]

    raw_reads = []
    candidates = []
    for name, variant, lang, config in variants:
        try:
            text = _ocr_text(variant, lang=lang, config=config)
            raw_reads.append({"source": name, "text": text})
            for value in _extract_candidates(text):
                if _national_id_structure_ok(value):
                    candidates.append({"value": value, "source": name})
        except Exception as exc:
            raw_reads.append({"source": name, "error": str(exc)})

    if not candidates:
        return {"value": None, "confidence": 0.0, "agreement": 0, "candidates": [], "raw_reads": raw_reads, "roi": roi}

    counts: dict[str, int] = {}
    for item in candidates:
        counts[item["value"]] = counts.get(item["value"], 0) + 1
    best_value = max(counts, key=counts.get)
    agreement = counts[best_value]
    confidence = min(0.99, 0.55 + agreement * 0.1)
    return {
        "value": best_value,
        "confidence": confidence,
        "agreement": agreement,
        "candidates": candidates,
        "raw_reads": raw_reads,
        "roi": roi,
    }


def extract_insurance_text(image: Image.Image) -> dict:
    enhanced = deskew_and_enhance(image)
    reads = []
    for lang in ("eng", "ara", "eng+ara"):
        try:
            text = _ocr_text(enhanced, lang=lang, config="--psm 6")
            reads.append({"lang": lang, "text": text})
        except Exception as exc:
            reads.append({"lang": lang, "error": str(exc), "text": ""})
    combined = "\n".join(item.get("text", "") for item in reads if item.get("text"))
    return {"lines": reads, "enhanced": enhanced, "text": combined}
