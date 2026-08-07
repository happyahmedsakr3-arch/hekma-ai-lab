import re
from functools import lru_cache
from typing import Any

import cv2
import numpy as np
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
    """Broad lower-right ROI for the Egyptian national ID number."""
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


@lru_cache(maxsize=2)
def _ocr(lang: str):
    from paddleocr import PaddleOCR

    return PaddleOCR(
        lang=lang,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="cpu",
    )


def _json_from_result(result: Any) -> dict:
    if isinstance(result, dict):
        return result
    data = getattr(result, "json", None)
    if callable(data):
        data = data()
    return data if isinstance(data, dict) else {}


def paddle_lines(image: Image.Image, lang: str = "ar") -> list[dict]:
    arr = np.array(image.convert("RGB"))
    outputs = _ocr(lang).predict(arr)
    lines: list[dict] = []

    for result in outputs:
        payload = _json_from_result(result)
        res = payload.get("res", payload)
        texts = list(res.get("rec_texts") or [])
        scores = list(res.get("rec_scores") or [])
        boxes = list(res.get("rec_boxes") or res.get("rec_polys") or [])
        for i, text in enumerate(texts):
            score = float(scores[i]) if i < len(scores) else 0.0
            box = boxes[i] if i < len(boxes) else None
            if hasattr(box, "tolist"):
                box = box.tolist()
            lines.append({"text": str(text), "score": score, "box": box})
    return lines


def _national_id_structure_ok(value: str) -> bool:
    if len(value) != 14 or value[0] not in "23":
        return False
    try:
        century = 1900 if value[0] == "2" else 2000
        year = century + int(value[1:3])
        month = int(value[3:5])
        day = int(value[5:7])
        import datetime as _dt
        _dt.date(year, month, day)
    except Exception:
        return False
    return True


def extract_national_id(image: Image.Image) -> dict:
    roi = national_id_roi(image)
    variants = [
        ("roi", roi),
        ("enhanced", deskew_and_enhance(roi)),
    ]
    candidates: list[dict] = []

    for name, variant in variants:
        for lang in ("ar", "en"):
            try:
                lines = paddle_lines(variant, lang=lang)
            except Exception as exc:
                candidates.append({"source": f"{name}:{lang}", "error": str(exc)})
                continue

            joined = " ".join(x["text"] for x in lines)
            raw_digit_groups = re.findall(r"[0-9٠-٩۰-۹][0-9٠-٩۰-۹\s.\-]{10,30}", joined)
            for group in raw_digit_groups:
                digits = normalize_digits(group)
                # Exact 14 is ideal; also inspect windows when OCR merged nearby digits.
                windows = [digits] if len(digits) == 14 else [digits[i:i+14] for i in range(max(0, len(digits)-13))]
                for value in windows:
                    if _national_id_structure_ok(value):
                        candidates.append({
                            "value": value,
                            "source": f"{name}:{lang}",
                            "confidence": max((x["score"] for x in lines if normalize_digits(x["text"]) in value or value in normalize_digits(x["text"])), default=0.0),
                        })

    good = [c for c in candidates if c.get("value")]
    if not good:
        return {"value": None, "confidence": 0.0, "candidates": candidates, "roi": roi}

    # Prefer agreement across independent language/processing passes.
    counts: dict[str, int] = {}
    for c in good:
        counts[c["value"]] = counts.get(c["value"], 0) + 1
    best = sorted(good, key=lambda c: (counts[c["value"]], c.get("confidence", 0.0)), reverse=True)[0]
    agreement = counts[best["value"]]
    confidence = min(0.99, max(best.get("confidence", 0.0), 0.70 + 0.08 * agreement))
    return {"value": best["value"], "confidence": confidence, "agreement": agreement, "candidates": good, "roi": roi}


def extract_insurance_text(image: Image.Image) -> dict:
    enhanced = deskew_and_enhance(image)
    all_lines: list[dict] = []
    for lang in ("en", "ar"):
        try:
            all_lines.extend({**line, "lang": lang} for line in paddle_lines(enhanced, lang=lang))
        except Exception:
            pass
    return {"lines": all_lines, "enhanced": enhanced, "text": "\n".join(line["text"] for line in all_lines)}
