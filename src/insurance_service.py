import base64
import io
import os
from datetime import date
from typing import Literal

from openai import OpenAI
from PIL import Image
from pydantic import BaseModel, Field

from src.insurance_parser import build_hints
from src.ocr_engine import extract_insurance_text


class FieldValue(BaseModel):
    value: str | None = None
    confidence: int = Field(default=0, ge=0, le=100)


class InsuranceResult(BaseModel):
    patient_name_ar: FieldValue
    patient_name_en: FieldValue
    insurance_company: FieldValue
    id_number: FieldValue
    member_id: FieldValue
    card_number: FieldValue
    policy_number: FieldValue
    employer: FieldValue
    network_class: FieldValue
    valid_from: FieldValue
    expiry_date: FieldValue
    card_status: Literal["valid", "expired", "unknown"]
    warnings: list[str]
    summary_ar: str


INSURANCE_PROMPT = """
أنت قارئ ومراجع لكارنيه تأمين طبي فقط.
سيصلك نص OCR وصورة كارنيه التأمين ومعهما HINTS مستخرجة من عناوين الحقول المطبوعة.

قواعد إلزامية:
- اقرأ بيانات كارنيه التأمين فقط. لا تبحث عن بطاقة رقم قومي ولا تحاول استنتاج بيانات غير موجودة.
- Card Number = فقط القيمة بجوار Card Number / Card No / Card #.
- ID No / ID Number = فقط القيمة بجوار ID No / ID Number / ID.
- Member ID = فقط إذا كان العنوان المطبوع Member ID / Membership ID.
- Policy Number = فقط إذا كان العنوان Policy No / Policy Number.
- Policy Holder = جهة العمل أو صاحب الوثيقة، وليس Policy Number.
- احتفظ بالحروف داخل أرقام الكارنيه كما هي مطبوعة.
- استخرج جهة العمل، الفئة/الشبكة، بداية الصلاحية وتاريخ الانتهاء إن وجدت.
- حدّد حالة الكارنيه مقارنة بتاريخ اليوم.
- HINTS ذات العنوان الواضح لها أولوية، واستخدم الصورة لحسم اللبس فقط.
- ممنوع اختراع رقم. إذا لم يوجد الحقل أو لم يكن واضحاً أرجع null.
"""


def open_image_bytes(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def image_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=94)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def review_insurance(image: Image.Image, ocr_text: str, hints: dict, api_key: str, model: str | None = None) -> InsuranceResult:
    if not api_key:
        raise RuntimeError("OpenAI API Key غير موجود")

    client = OpenAI(api_key=api_key)
    response = client.responses.parse(
        model=model or os.getenv("OPENAI_MODEL", "gpt-5.1"),
        instructions=INSURANCE_PROMPT,
        input=[{
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": f"تاريخ اليوم: {date.today().isoformat()}\n\nHINTS:\n{hints}\n\nOCR TEXT:\n{ocr_text}",
                },
                {
                    "type": "input_image",
                    "image_url": image_url(image),
                    "detail": "high",
                },
            ],
        }],
        text_format=InsuranceResult,
    )

    if response.output_parsed is None:
        raise RuntimeError("تعذر تنظيم بيانات الكارنيه")
    return response.output_parsed


def analyze_insurance_card(data: bytes, api_key: str, model: str | None = None) -> dict:
    image = open_image_bytes(data)
    ocr = extract_insurance_text(image)
    hints = build_hints(ocr.get("text", ""))
    result = review_insurance(image, ocr.get("text", ""), hints, api_key=api_key, model=model)
    return {
        "result": result.model_dump(),
        "ocr": {
            "text": ocr.get("text", ""),
            "lines": ocr.get("lines", []),
        },
        "hints": hints,
    }


def collector_payload(bundle: dict) -> dict:
    r = bundle["result"]
    hints = bundle.get("hints", {})

    def val(name: str):
        field = r.get(name) or {}
        return field.get("value") if isinstance(field, dict) else None

    return {
        "ok": True,
        "patient_name": val("patient_name_en") or val("patient_name_ar") or hints.get("name") or "",
        "patient_name_ar": val("patient_name_ar") or "",
        "patient_name_en": val("patient_name_en") or "",
        "insurance_company": val("insurance_company") or hints.get("company") or "",
        "card_number": val("card_number") or hints.get("card_number") or "",
        "id_number": val("id_number") or hints.get("id_number") or "",
        "member_id": val("member_id") or hints.get("member_id") or "",
        "policy_number": val("policy_number") or hints.get("policy_number") or "",
        "employer": val("employer") or hints.get("employer") or "",
        "network_class": val("network_class") or hints.get("network_class") or "",
        "valid_from": val("valid_from") or hints.get("valid_from") or "",
        "expiry_date": val("expiry_date") or hints.get("expiry_date") or "",
        "card_status": r.get("card_status", "unknown"),
        "warnings": r.get("warnings", []),
        "summary_ar": r.get("summary_ar", ""),
    }
