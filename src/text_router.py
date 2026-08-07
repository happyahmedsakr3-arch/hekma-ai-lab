import re
from PIL import Image

from src.ocr_engine import extract_insurance_text

INSURANCE_WORDS = {
    "globemed": 8,
    "globe med": 8,
    "axa": 8,
    "nextcare": 8,
    "next care": 8,
    "metlife": 8,
    "medright": 8,
    "med right": 8,
    "mednet": 8,
    "bupa": 8,
    "healthcare membership": 6,
    "insurance card": 6,
    "card number": 5,
    "card no": 4,
    "member id": 5,
    "membership id": 5,
    "policy no": 5,
    "policy number": 5,
    "policy holder": 4,
    "valid until": 4,
    "valid to": 4,
    "expiry": 4,
    "network": 3,
    "category": 3,
}

ID_WORDS = {
    "جمهورية مصر العربية": 9,
    "بطاقة تحقيق الشخصية": 9,
    "تحقيق الشخصية": 7,
    "الرقم القومي": 7,
    "محل الإقامة": 3,
    "العنوان": 2,
    "المنوفية": 1,
    "القليوبية": 1,
    "القاهرة": 1,
    "الجيزة": 1,
}

ARABIC_TO_LATIN = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def _norm(text: str) -> str:
    return " ".join((text or "").lower().replace("_", " ").split())


def _valid_national_id_candidates(text: str) -> list[str]:
    normalized = (text or "").translate(ARABIC_TO_LATIN)
    candidates = []
    for chunk in re.findall(r"[0-9][0-9\s.\-]{10,32}", normalized):
        digits = re.sub(r"\D", "", chunk)
        windows = [digits] if len(digits) == 14 else [digits[i:i + 14] for i in range(max(0, len(digits) - 13))]
        for value in windows:
            if len(value) != 14 or value[0] not in "23":
                continue
            try:
                import datetime as dt
                century = 1900 if value[0] == "2" else 2000
                born = dt.date(century + int(value[1:3]), int(value[3:5]), int(value[5:7]))
                if born <= dt.date.today():
                    candidates.append(value)
            except Exception:
                pass
    return candidates


def score_text(text: str) -> dict:
    low = _norm(text)
    insurance_score = 0
    id_score = 0
    insurance_hits = []
    id_hits = []

    for word, weight in INSURANCE_WORDS.items():
        if word in low:
            insurance_score += weight
            insurance_hits.append(word)

    for word, weight in ID_WORDS.items():
        if word in (text or ""):
            id_score += weight
            id_hits.append(word)

    national_ids = _valid_national_id_candidates(text)
    if national_ids:
        id_score += 10
        id_hits.append("14-digit national-id pattern")

    # Insurance cards very often contain labelled alphanumeric identifiers.
    if re.search(r"card\s*(?:number|no)\s*[:\-]?\s*[A-Z0-9\-]{5,}", low, re.I):
        insurance_score += 5
    if re.search(r"\b(?:id|member\s*id)\s*[:\-]?\s*\d{5,}", low, re.I):
        insurance_score += 3

    if insurance_score >= 8 and id_score >= 8:
        kind = "mixed"
    elif insurance_score >= max(6, id_score + 2):
        kind = "insurance"
    elif id_score >= max(6, insurance_score + 2):
        kind = "national_id"
    else:
        kind = "unknown"

    return {
        "kind": kind,
        "insurance_score": insurance_score,
        "id_score": id_score,
        "insurance_hits": insurance_hits,
        "id_hits": id_hits,
        "national_id_candidates": national_ids,
    }


def _segments(image: Image.Image) -> list[tuple[str, Image.Image]]:
    image = image.convert("RGB")
    w, h = image.size
    segments = [("full", image)]

    # Only split images large enough for each resulting half to remain readable.
    if h >= 600:
        segments.extend([
            ("top", image.crop((0, 0, w, int(h * 0.56)))),
            ("bottom", image.crop((0, int(h * 0.44), w, h))),
        ])
    if w >= 900:
        segments.extend([
            ("left", image.crop((0, 0, int(w * 0.56), h))),
            ("right", image.crop((int(w * 0.44), 0, w, h))),
        ])
    return segments


def route_image_by_text(image: Image.Image) -> dict:
    candidates = []
    for label, segment in _segments(image):
        ocr = extract_insurance_text(segment)
        scores = score_text(ocr.get("text", ""))
        candidates.append({
            "label": label,
            "image": segment,
            "ocr": ocr,
            **scores,
        })

    best_insurance = max(candidates, key=lambda x: x["insurance_score"])
    best_id = max(candidates, key=lambda x: x["id_score"])

    insurance_ok = best_insurance["insurance_score"] >= 6
    id_ok = best_id["id_score"] >= 6

    return {
        "segments": candidates,
        "insurance_image": best_insurance["image"] if insurance_ok else None,
        "insurance_ocr": best_insurance["ocr"] if insurance_ok else None,
        "insurance_segment": best_insurance["label"] if insurance_ok else None,
        "insurance_score": best_insurance["insurance_score"],
        "id_image": best_id["image"] if id_ok else None,
        "id_segment": best_id["label"] if id_ok else None,
        "id_score": best_id["id_score"],
        "full_kind": candidates[0]["kind"],
    }
