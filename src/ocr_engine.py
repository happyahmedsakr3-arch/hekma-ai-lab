from typing import Any

import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def _prepare(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    w, h = image.size
    if max(w, h) < 1800:
        scale = 1800 / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    gray = ImageEnhance.Contrast(gray).enhance(1.7)
    gray = ImageEnhance.Sharpness(gray).enhance(2.0)
    gray = gray.filter(ImageFilter.UnsharpMask(radius=1.4, percent=170, threshold=3))
    return gray.convert("RGB")


def _ocr(image: Image.Image, lang: str, config: str) -> str:
    return pytesseract.image_to_string(image, lang=lang, config=config)


def extract_insurance_text(image: Image.Image) -> dict[str, Any]:
    enhanced = _prepare(image)
    reads: list[dict[str, Any]] = []

    for lang, config in (
        ("eng", "--psm 6"),
        ("eng+ara", "--psm 6"),
        ("eng", "--psm 11"),
    ):
        try:
            text = _ocr(enhanced, lang, config)
            reads.append({"lang": lang, "config": config, "text": text})
        except Exception as exc:
            reads.append({"lang": lang, "config": config, "text": "", "error": str(exc)})

    texts = [item["text"] for item in reads if item.get("text")]
    combined = "\n\n--- OCR PASS ---\n\n".join(texts)

    return {
        "text": combined,
        "lines": reads,
        "enhanced": enhanced,
    }
