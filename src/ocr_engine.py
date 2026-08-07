import re
from typing import Any

import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

ARABIC_TO_LATIN = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def normalize_digits(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\D", "", text.translate(ARABIC_TO_LATIN))


def deskew_and_enhance(image: Image.Image) -> Image.Image:
    """Lightweight server-safe preprocessing using Pillow only."""
    gray = ImageOps.grayscale(image.convert("RGB"))
    gray = ImageOps.autocontrast(gray, cutoff=1)
    gray = ImageEnhance.Contrast(gray).enhance(1.7)
    gray = ImageEnhance.Sharpness(gray).enhance(2.0)
    gray = gray.filter(ImageFilter.UnsharpMask(radius=1.4, percent=170, threshold=3))
    return gray.convert("RGB")


def national_id_roi(image: Image.Image) -> Image.Image:
    """Broad lower-right ROI for the Egyptian national-ID number."""
    image = image.convert("RGB")
    w, h = image.size
    x1, y1 = int(w * 0.30), int(h * 0.58)
    x2, y2 = int(w * 0.995), int(h * 0.96)
    roi = image.crop((x1, y1, x2, y2))
    if roi.width < 5 or roi.height < 5:
        return image

    target_w = max(2200, roi.width * 5)
    scale = target_w / roi.width
    target_h = max(350, int(roi.height * scale))
    roi = roi.resize((target_w, target_h), Image.Resampling.LANCZOS)
    gray = ImageOps.grayscale(roi)
    gray = ImageOps.autocontrast(gray, cutoff=0.5)
    gray = ImageEnhance.Contrast(gray).enhance(2.3)
    gray = ImageEnhance.Sharpness(gray).enhance(2.8)
    gray = gray.filter(ImageFilter.MedianFilter(size=3))
    gray = gray.filter(ImageFilter.UnsharpMask(radius=1.6, percent=220, threshold=2))
    return gray.convert("RGB")


def _threshold_variant(image: Image.Image, level: int) -> Image.Image:
    gray = ImageOps.grayscale(image)
    return gray.point(lambda p: 255 if p > level else 0).convert("RGB")


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
        birth = _dt.date(year, month, day)
        if birth > _dt.date.today():
            return False
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
                candidates.append(digits[i:i + 14])
    return candidates


def extract_national_id(image: Image.Image) -> dict:
    roi = national_id_roi(image)
    threshold_135 = _threshold_variant(roi, 135)
    threshold_165 = _threshold_variant(roi, 165)

    variants = [
        ("roi_ara", roi, "ara", "--psm 6"),
        ("roi_eng", roi, "eng", "--psm 6"),
        ("t135_ara", threshold_135, "ara", "--psm 6"),
        ("t165_ara", threshold_165, "ara", "--psm 6"),
        ("digits", roi, "eng", "--psm 7 -c tessedit_char_whitelist=0123456789"),
    ]

    raw_reads: list[dict[str, Any]] = []
    candidates: list[dict[str, str]] = []
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
        return {
            "value": None,
            "confidence": 0.0,
            "agreement": 0,
            "candidates": [],
            "raw_reads": raw_reads,
            "roi": roi,
        }

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
    reads: list[dict[str, Any]] = []
    for lang in ("eng", "ara", "eng+ara"):
        try:
            text = _ocr_text(enhanced, lang=lang, config="--psm 6")
            reads.append({"lang": lang, "text": text})
        except Exception as exc:
            reads.append({"lang": lang, "error": str(exc), "text": ""})
    combined = "\n".join(item.get("text", "") for item in reads if item.get("text"))
    return {"lines": reads, "enhanced": enhanced, "text": combined}
