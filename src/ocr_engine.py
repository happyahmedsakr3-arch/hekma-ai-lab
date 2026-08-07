import re
from typing import Any

import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

ARABIC_TO_LATIN = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def normalize_digits(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\D", "", text.translate(ARABIC_TO_LATIN))


def _resize_for_ocr(image: Image.Image, min_width: int = 1800) -> Image.Image:
    image = image.convert("RGB")
    if image.width >= min_width:
        return image
    scale = min_width / max(1, image.width)
    return image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)


def deskew_and_enhance(image: Image.Image) -> Image.Image:
    image = _resize_for_ocr(image)
    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray, cutoff=0.5)
    gray = ImageEnhance.Contrast(gray).enhance(1.8)
    gray = ImageEnhance.Sharpness(gray).enhance(2.1)
    gray = gray.filter(ImageFilter.UnsharpMask(radius=1.3, percent=180, threshold=2))
    return gray.convert("RGB")


def national_id_roi(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    w, h = image.size
    x1, y1 = int(w * 0.22), int(h * 0.52)
    x2, y2 = int(w * 0.995), int(h * 0.98)
    roi = image.crop((x1, y1, x2, y2))
    if roi.width < 5 or roi.height < 5:
        return image
    target_w = max(2400, roi.width * 6)
    scale = target_w / roi.width
    target_h = max(450, int(roi.height * scale))
    roi = roi.resize((target_w, target_h), Image.Resampling.LANCZOS)
    gray = ImageOps.grayscale(roi)
    gray = ImageOps.autocontrast(gray, cutoff=0.3)
    gray = ImageEnhance.Contrast(gray).enhance(2.0)
    gray = ImageEnhance.Sharpness(gray).enhance(2.5)
    gray = gray.filter(ImageFilter.UnsharpMask(radius=1.2, percent=210, threshold=1))
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
        birth = _dt.date(century + int(value[1:3]), int(value[3:5]), int(value[5:7]))
        if birth > _dt.date.today():
            return False
    except Exception:
        return False
    return True


def _extract_candidates(text: str) -> list[str]:
    normalized = (text or "").translate(ARABIC_TO_LATIN)
    candidates: list[str] = []
    for group in re.findall(r"[0-9][0-9\s.\-]{10,36}", normalized):
        digits = re.sub(r"\D", "", group)
        if len(digits) == 14:
            candidates.append(digits)
        elif len(digits) > 14:
            for i in range(len(digits) - 13):
                candidates.append(digits[i:i + 14])
    return candidates


def extract_national_id(image: Image.Image) -> dict:
    roi = national_id_roi(image)
    t120 = _threshold_variant(roi, 120)
    t145 = _threshold_variant(roi, 145)
    t170 = _threshold_variant(roi, 170)

    variants = [
        ("roi_ara_p6", roi, "ara", "--psm 6"),
        ("roi_ara_p11", roi, "ara", "--psm 11"),
        ("roi_eng_p6", roi, "eng", "--psm 6"),
        ("roi_eng_p11", roi, "eng", "--psm 11"),
        ("t120_ara", t120, "ara", "--psm 6"),
        ("t145_ara", t145, "ara", "--psm 6"),
        ("t170_ara", t170, "ara", "--psm 6"),
        ("digits_p7", roi, "eng", "--psm 7 -c tessedit_char_whitelist=0123456789"),
        ("digits_p11", roi, "eng", "--psm 11 -c tessedit_char_whitelist=0123456789"),
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
        return {"value": None, "confidence": 0.0, "agreement": 0, "candidates": [], "raw_reads": raw_reads, "roi": roi}

    counts: dict[str, int] = {}
    for item in candidates:
        counts[item["value"]] = counts.get(item["value"], 0) + 1
    best_value = max(counts, key=counts.get)
    agreement = counts[best_value]
    confidence = min(0.99, 0.50 + agreement * 0.1)
    return {"value": best_value, "confidence": confidence, "agreement": agreement, "candidates": candidates, "raw_reads": raw_reads, "roi": roi}


def _unique_join(texts: list[str]) -> str:
    seen = set()
    out = []
    for text in texts:
        clean = (text or "").strip()
        if clean and clean not in seen:
            seen.add(clean)
            out.append(clean)
    return "\n\n--- OCR PASS ---\n\n".join(out)


def extract_insurance_text(image: Image.Image) -> dict:
    original = _resize_for_ocr(image)
    enhanced = deskew_and_enhance(image)
    t135 = _threshold_variant(enhanced, 135)
    t165 = _threshold_variant(enhanced, 165)

    passes = [
        ("original_eng_p6", original, "eng", "--psm 6"),
        ("original_eng_p11", original, "eng", "--psm 11"),
        ("original_ara_p6", original, "ara", "--psm 6"),
        ("enhanced_eng_p6", enhanced, "eng", "--psm 6"),
        ("enhanced_eng_p11", enhanced, "eng", "--psm 11"),
        ("enhanced_ara_p6", enhanced, "ara", "--psm 6"),
        ("enhanced_mix_p6", enhanced, "eng+ara", "--psm 6"),
        ("t135_eng", t135, "eng", "--psm 6"),
        ("t165_eng", t165, "eng", "--psm 6"),
    ]

    reads: list[dict[str, Any]] = []
    texts: list[str] = []
    for name, variant, lang, config in passes:
        try:
            text = _ocr_text(variant, lang=lang, config=config)
            reads.append({"source": name, "lang": lang, "text": text})
            if text.strip():
                texts.append(text)
        except Exception as exc:
            reads.append({"source": name, "lang": lang, "error": str(exc), "text": ""})

    return {"lines": reads, "enhanced": enhanced, "text": _unique_join(texts)}
