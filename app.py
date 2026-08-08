import base64
import io
import os
from datetime import date
from typing import Literal

import streamlit as st
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


def api_key() -> str:
    try:
        return st.secrets.get("OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
    except Exception:
        return os.getenv("OPENAI_API_KEY", "")


def open_upload(file) -> Image.Image:
    return Image.open(io.BytesIO(file.getvalue())).convert("RGB")


def image_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=94)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def review_insurance(image: Image.Image, ocr_text: str, hints: dict) -> InsuranceResult:
    if not api_key():
        raise RuntimeError("OpenAI API Key غير موجود")

    client = OpenAI(api_key=api_key())
    response = client.responses.parse(
        model=os.getenv("OPENAI_MODEL", "gpt-5.1"),
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


st.set_page_config(page_title="Hekma AI - Insurance Card Reader", page_icon="🧠", layout="wide")
st.markdown(
    """
    <style>
    html,body,[class*='css']{direction:rtl;text-align:right}
    .block-container{max-width:1100px;padding-top:2rem}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧠 Hekma AI — قارئ كارنيه التأمين")
st.caption("ارفع صورة كارنيه التأمين فقط، وسيتم استخراج بياناته ومراجعتها.")

with st.sidebar:
    st.info("الوضع الحالي: قارئ كارنيه التأمين فقط")
    st.success("GPT متصل") if api_key() else st.warning("GPT غير متصل")

insurance_file = st.file_uploader(
    "ارفع صورة كارنيه التأمين",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=False,
    key="insurance_card_only",
)

if insurance_file:
    st.image(insurance_file, caption="كارنيه التأمين", use_container_width=True)

if st.button("تحليل الكارنيه", type="primary", use_container_width=True):
    if not insurance_file:
        st.error("ارفع صورة كارنيه التأمين أولاً")
    else:
        try:
            with st.spinner("جاري قراءة الكارنيه وتدقيق البيانات..."):
                image = open_upload(insurance_file)
                ocr = extract_insurance_text(image)
                hints = build_hints(ocr.get("text", ""))
                result = review_insurance(image, ocr.get("text", ""), hints) if api_key() else None

                st.session_state.card_ocr = ocr
                st.session_state.card_hints = hints
                st.session_state.card_result = result
        except Exception as exc:
            st.error(f"فشل تحليل الكارنيه: {exc}")

ocr = st.session_state.get("card_ocr")
hints = st.session_state.get("card_hints", {})
result = st.session_state.get("card_result")

if ocr:
    tab1, tab2, tab3 = st.tabs(["النتيجة", "OCR الخام", "التدقيق"])

    with tab1:
        st.subheader("بيانات التأمين")

        if result:
            st.success(result.summary_ar)
            a, b = st.columns(2)
            a.text_input("الاسم", value=result.patient_name_en.value or result.patient_name_ar.value or hints.get("name") or "")
            b.text_input("شركة التأمين", value=result.insurance_company.value or hints.get("company") or "")
            a.text_input("Card Number", value=result.card_number.value or hints.get("card_number") or "")
            b.text_input("ID No", value=result.id_number.value or hints.get("id_number") or "")
            a.text_input("Member ID", value=result.member_id.value or hints.get("member_id") or "")
            b.text_input("Policy Number", value=result.policy_number.value or hints.get("policy_number") or "")
            a.text_input("جهة العمل / Policy Holder", value=result.employer.value or hints.get("employer") or "")
            b.text_input("الفئة / الشبكة", value=result.network_class.value or hints.get("network_class") or "")
            a.text_input("بداية الصلاحية", value=result.valid_from.value or hints.get("valid_from") or "")
            b.text_input("تاريخ الانتهاء", value=result.expiry_date.value or hints.get("expiry_date") or "")

            status = {
                "valid": "🟢 الكارنيه ساري",
                "expired": "🔴 الكارنيه منتهي",
                "unknown": "🟡 حالة الكارنيه غير مؤكدة",
            }[result.card_status]
            st.subheader(status)
        else:
            st.info("تمت قراءة OCR. أضف OpenAI API Key لعرض الحقول المنظمة.")
            if hints:
                st.json(hints)

    with tab2:
        st.text_area("نص OCR", value=ocr.get("text", ""), height=320)

    with tab3:
        if hints:
            st.write("الحقول التي استخرجتها القواعد قبل GPT:")
            st.json(hints)
        if result and result.warnings:
            st.warning("\n".join("• " + x for x in result.warnings))
