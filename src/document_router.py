import re
from PIL import Image

from src.ocr_engine import extract_insurance_text

ARABIC_TO_LATIN = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

INSURANCE_MARKERS = (
    "card number", "member id", "membership", "policy no", "policy holder",
    "valid until", "valid to", "expiry", "insurance card", "globemed",
    "nextcare", "axa", "metlife", "medright", "bupa", "network", "category",
)

NATIONAL_ID_MARKERS = (
    "بطاقة تحقيق الشخصية", "جمهورية مصر العربية", "الرقم القومي",
    "مركز", "قسم", "محافظة",
)


def _norm(text: str) -> str:
    return (text or "").lower().translate(ARABIC_TO_LATIN)


def _has_14_digits(text: str) -> bool:
    text = _norm(text)
    groups = re.findall(r"[0-9][0-9\s.\-]{10,30}", text)
    for group in groups:
        digits = re.sub(r"\D", "", group)
        if len(digits) == 14 and digits[0] in "23":
            return True
    return False


def classify_text(text: str) -> dict:
    t = _norm(text)
    insurance_score = sum(1 for m in INSURANCE_MARKERS if m in t)
    id_score = sum(1 for m in NATIONAL_ID_MARKERS if m in t)
    if _has_14_digits(t):
        id_score += 3

    if insurance_score >= 2 and id_score >= 2:
        kind = "combined"
    elif insurance_score >= max(2, id_score + 1):
        kind = "insurance"
    elif id_score >= max(2, insurance_score + 1):
        kind = "national_id"
    else:
        kind = "unknown"

    return {
        "kind": kind,
        "insurance_score": insurance_score,
        "id_score": id_score,
    }


def classify_image(image: Image.Image) -> dict:
    ocr = extract_insurance_text(image)
    result = classify_text(ocr.get("text", ""))
    result["ocr"] = ocr
    return result


def split_candidates(image: Image.Image) -> list[tuple[str, Image.Image]]:
    w, h = image.size
    return [
        ("top", image.crop((0, 0, w, h // 2))),
        ("bottom", image.crop((0, h // 2, w, h))),
        ("left", image.crop((0, 0, w // 2, h))),
        ("right", image.crop((w // 2, 0, w, h))),
    ]


def route_image(image: Image.Image) -> dict:
    full = classify_image(image)
    if full["kind"] != "combined":
        return {
            "kind": full["kind"],
            "insurance_image": image if full["kind"] == "insurance" else None,
            "id_image": image if full["kind"] == "national_id" else None,
            "full": full,
            "parts": [],
        }

    parts = []
    for name, part in split_candidates(image):
        classified = classify_image(part)
        parts.append({"name": name, "image": part, **classified})

    insurance_parts = [p for p in parts if p["kind"] == "insurance"]
    id_parts = [p for p in parts if p["kind"] == "national_id"]

    insurance_image = None
    id_image = None
    if insurance_parts:
        insurance_image = max(insurance_parts, key=lambda p: p["insurance_score"])["image"]
    if id_parts:
        id_image = max(id_parts, key=lambda p: p["id_score"])["image"]

    return {
        "kind": "combined",
        "insurance_image": insurance_image,
        "id_image": id_image,
        "full": full,
        "parts": parts,
    }
