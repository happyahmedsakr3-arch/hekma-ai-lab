import re
from functools import lru_cache
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from paddleocr import PaddleOCR

ARABIC_TO_LATIN = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def normalize_digits(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\D", "", text.translate(ARABIC_TO_LATIN))


@lru_cache(maxsize=1)
def _ocr() -> PaddleOCR:
    return PaddleOCR(
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="arabic_PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="cpu",
    )


def _prepare(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    w, h = image.size
    if max(w, h) < 1800:
        scale = 1800 / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    image = ImageOps.autocontrast(image, cutoff=0.5)
    image = ImageEnhance.Contrast(image).enhance(1.25)
    image = ImageEnhance.Sharpness(image).enhance(1.35)
    image = image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=140, threshold=2))
    return image


def _run(image: Image.Image) -> dict[str, Any]:
    prepared = _prepare(image)
    arr = np.array(prepared)
    outputs = list(
        _ocr().predict(
            arr,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_rec_score_thresh=0.25,
        )
    )

    texts: list[str] = []
    scores: list[float] = []
    boxes: list[Any] = []
    raw: list[dict[str, Any]] = []

    for out in outputs:
        payload = out.json
        res = payload.get("res", payload)
        part_texts = [str(x) for x in (res.get("rec_texts") or []) if str(x).strip()]
        part_scores = [float(x) for x in (res.get("rec_scores") or [])]
        part_boxes = res.get("rec_boxes") or []
        texts.extend(part_texts)
        scores.extend(part_scores[: len(part_texts)])
        try:
            boxes.extend(part_boxes.tolist())
        except Exception:
            boxes.extend(part_boxes)
        raw.append({
            "texts": part_texts,
            "scores": part_scores,
        })

    return {
        "text": "\n".join(texts),
        "texts": texts,
        "scores": scores,
        "boxes": boxes,
        "raw": raw,
        "enhanced": prepared,
    }


def extract_insurance_text(image: Image.Image) -> dict:
    result = _run(image)
    lines = []
    for idx, text in enumerate(result["texts"]):
        score = result["scores"][idx] if idx < len(result["scores"]) else None
        lines.append({"text": text, "confidence": score})
    result["lines"] = lines
    return result


def _national_id_structure_ok(value: str) -> bool:
    if len(value) != 14 or value[0] not in "23":
        return False
    try:
        import datetime as dt
        century = 1900 if value[0] == "2" else 2000
        born = dt.date(century + int(value[1:3]), int(value[3:5]), int(value[5:7]))
        return born <= dt.date.today()
    except Exception:
        return False


def _extract_candidates(text: str) -> list[str]:
    normalized = (text or "").translate(ARABIC_TO_LATIN)
    candidates: list[str] = []
    for chunk in re.findall(r"[0-9][0-9\s.\-]{10,40}", normalized):
        digits = re.sub(r"\D", "", chunk)
        if len(digits) == 14:
            candidates.append(digits)
        elif len(digits) > 14:
            for i in range(len(digits) - 13):
                candidates.append(digits[i:i + 14])
    return candidates


def national_id_roi(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    w, h = image.size
    # Egyptian ID number is normally in the lower half; keep a broad ROI to tolerate perspective/crops.
    roi = image.crop((int(w * 0.18), int(h * 0.50), int(w * 0.995), int(h * 0.98)))
    if roi.width < 10 or roi.height < 10:
        return image
    target_w = max(2200, roi.width * 4)
    scale = target_w / roi.width
    return roi.resize((target_w, int(roi.height * scale)), Image.Resampling.LANCZOS)


def extract_national_id(image: Image.Image) -> dict:
    roi = national_id_roi(image)
    variants = [
        ("roi", roi),
        ("full", image),
        ("roi_contrast", ImageEnhance.Contrast(ImageOps.autocontrast(roi.convert("RGB"))).enhance(1.5)),
    ]

    raw_reads: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    for name, variant in variants:
        try:
            out = _run(variant)
            raw_reads.append({"source": name, "text": out["text"]})
            for value in _extract_candidates(out["text"]):
                if _national_id_structure_ok(value):
                    # confidence = best OCR score on this pass, not fake certainty
                    conf = max(out.get("scores") or [0.0])
                    candidates.append({"value": value, "source": name, "ocr_confidence": conf})
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
    best_conf: dict[str, float] = {}
    for item in candidates:
        value = item["value"]
        counts[value] = counts.get(value, 0) + 1
        best_conf[value] = max(best_conf.get(value, 0.0), float(item.get("ocr_confidence", 0.0)))

    best_value = max(counts, key=lambda v: (counts[v], best_conf.get(v, 0.0)))
    agreement = counts[best_value]
    confidence = best_conf.get(best_value, 0.0)
    return {
        "value": best_value,
        "confidence": confidence,
        "agreement": agreement,
        "candidates": candidates,
        "raw_reads": raw_reads,
        "roi": roi,
    }
