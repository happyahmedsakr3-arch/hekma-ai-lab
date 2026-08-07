import re
from PIL import Image

from src.paddle_engine import extract_insurance_text

INSURANCE_WORDS = {
    "globemed": 10,
    "globe med": 10,
    "axa": 10,
    "nextcare": 10,
    "next care": 10,
    "metlife": 10,
    "medright": 10,
    "med right": 10,
    "mednet": 10,
    "bupa": 10,
    "healthcare membership": 7,
    "insurance card": 7,
    "card number": 6,
    "card no": 5,
    "member id": 6,
    "membership id": 6,
    "policy no": 6,
    "policy number": 6,
    "policy holder": 5,
    "valid until": 5,
    "valid to": 5,
    "expiry": 5,
    "network": 3,
    "category": 3,
}

ID_WORDS = {
    "جمهورية مصر العربية": 10,
    "بطاقة تحقيق الشخصية": 10,
    "تحقيق الشخصية": 8,
    "الرقم القومي": 8,
    "بطاقة شخصية": 6,
    "محل الإقامة": 3,
    "العنوان": 2,
    "القليوبية": 1,
    "القاهرة": 1,
    "الجيزة": 1,
    "المنوفية": 1,
}

ARABIC_TO_LATIN = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def _norm(text: str) -> str:
    return " ".join((text or "").lower().replace("_", " ").split())


def _valid_national_id_candidates(text: str) -> list[str]:
    normalized = (text or "").translate(ARABIC_TO_LATIN)
    candidates = []
    for chunk in re.findall(r"[0-9][0-9\s.\-]{10,36}", normalized):
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
    return list(dict.fromkeys(candidates))


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
        id_score += 12
        id_hits.append("14-digit national-id pattern")

    if re.search(r"\b[A-Z]{1,3}\d{5,12}[A-Z0-9]*\b", text or "", re.I):
        insurance_score += 3
        insurance_hits.append("alphanumeric card pattern")
    if re.search(r"\b\d{7,9}\b", text or ""):
        insurance_score += 2
        insurance_hits.append("7-9 digit id pattern")
    if re.search(r"(?:valid|expiry|until|to).{0,20}\d{1,2}[\-/][A-Za-z0-9]{1,4}[\-/]\d{2,4}", low, re.I):
        insurance_score += 4
        insurance_hits.append("validity date pattern")

    if insurance_score >= 5 and id_score >= 7:
        kind = "mixed"
    elif insurance_score >= max(4, id_score + 1):
        kind = "insurance"
    elif id_score >= max(5, insurance_score + 1):
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

    if h >= 500:
        segments.extend([
            ("top", image.crop((0, 0, w, int(h * 0.58)))),
            ("bottom", image.crop((0, int(h * 0.42), w, h))),
        ])
    if w >= 700:
        segments.extend([
            ("left", image.crop((0, 0, int(w * 0.58), h))),
            ("right", image.crop((int(w * 0.42), 0, w, h))),
        ])
    return segments


def route_image_by_text(image: Image.Image) -> dict:
    candidates = []
    for label, segment in _segments(image):
        ocr = extract_insurance_text(segment)
        scores = score_text(ocr.get("text", ""))
        candidates.append({"label": label, "image": segment, "ocr": ocr, **scores})

    best_insurance = max(candidates, key=lambda x: x["insurance_score"])
    best_id = max(candidates, key=lambda x: x["id_score"])

    insurance_ok = best_insurance["insurance_score"] >= 4
    id_ok = best_id["id_score"] >= 5

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
