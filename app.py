import base64
import io
import os
import re
from datetime import date
from typing import Literal

import streamlit as st
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel, Field

from src.document_router import route_image
from src.insurance_parser import build_hints
from src.ocr_engine import extract_insurance_text, extract_national_id


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
أنت مراجع بيانات كارنيه تأمين طبي، وليس OCR.
سيصلك نص Tesseract وصورة تم تصنيفها أولاً ككارنيه تأمين، ومعهما HINTS من عناوين الحقول المطبوعة.
- لا تستخدم بيانات بطاقة الرقم القومي لملء Card Number أو Member ID أو Policy Number.
- Card Number = فقط القيمة بجوار Card Number / Card No.
- ID No / ID Number = حصراً في id_number.
- Member ID = فقط إذا كان العنوان Member ID / Membership ID.
- Policy Number = فقط إذا كان العنوان Policy No / Policy Number.
- Policy Holder = جهة العمل/صاحب الوثيقة وليس Policy Number.
- لا تخترع أي رقم. إذا لم يوجد الحقل صراحة أرجع null.
"""

ARABIC_TO_LATIN = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


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


def normalize_digits(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\D", "", value.translate(ARABIC_TO_LATIN))


def valid_egyptian_national_id(value: str | None) -> str | None:
    digits = normalize_digits(value)
    if len(digits) != 14 or digits[0] not in {"2", "3"}:
        return None
    try:
        century = 1900 if digits[0] == "2" else 2000
        born = date(century + int(digits[1:3]), int(digits[3:5]), int(digits[5:7]))
        if born > date.today():
            return None
    except ValueError:
        return None
    return digits


def review_insurance(image: Image.Image, ocr_text: str, hints: dict) -> InsuranceResult:
    if not api_key():
        raise RuntimeError("OpenAI API Key غير موجود")
    client = OpenAI(api_key=api_key())
    content = [
        {"type": "input_text", "text": f"تاريخ اليوم: {date.today().isoformat()}\n\nHINTS:\n{hints}\n\nOCR TEXT:\n{ocr_text}"},
        {"type": "input_image", "image_url": image_url(image), "detail": "high"},
    ]
    response = client.responses.parse(
        model=os.getenv("OPENAI_MODEL", "gpt-5.1"),
        instructions=INSURANCE_PROMPT,
        input=[{"role": "user", "content": content}],
        text_format=InsuranceResult,
    )
    if response.output_parsed is None:
        raise RuntimeError("تعذر تنظيم بيانات التأمين")
    return response.output_parsed


st.set_page_config(page_title="Hekma AI V4", page_icon="🧠", layout="wide")
st.markdown("""
<style>
html,body,[class*='css']{direction:rtl;text-align:right}
.block-container{max-width:1180px;padding-top:2rem}
</style>
""", unsafe_allow_html=True)

st.title("🧠 Hekma AI — Document AI V4")
st.caption("ارفع الصور بأي ترتيب. النظام يحدد تلقائياً: بطاقة رقم قومي / كارنيه تأمين / صورة تحتوي الاثنين.")

with st.sidebar:
    st.info("OCR مجاني: Tesseract عربي/إنجليزي")
    st.success("GPT متصل") if api_key() else st.warning("GPT غير متصل")

files = st.file_uploader(
    "ارفع صور المستندات — الترتيب غير مهم",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True,
    key="documents_v4",
)

if files:
    cols = st.columns(min(3, len(files)))
    for i, file in enumerate(files):
        cols[i % len(cols)].image(file, caption=file.name, use_container_width=True)

if st.button("تحليل المستندات تلقائياً", type="primary", use_container_width=True):
    if not files:
        st.error("ارفع صورة واحدة على الأقل")
    else:
        try:
            with st.spinner("جاري تصنيف الصور أولاً ثم تشغيل القارئ المناسب لكل مستند..."):
                routed = []
                insurance_images = []
                id_images = []

                for file in files:
                    image = open_upload(file)
                    route = route_image(image)
                    routed.append({"name": file.name, "route": route})
                    if route.get("insurance_image") is not None:
                        insurance_images.append((file.name, route["insurance_image"]))
                    if route.get("id_image") is not None:
                        id_images.append((file.name, route["id_image"]))

                # اختر أوضح نتيجة حسب درجات التصنيف، وليس ترتيب الرفع.
                insurance_img = insurance_images[0][1] if insurance_images else None
                id_img = id_images[0][1] if id_images else None

                insurance_ocr = insurance_result = national_result = hints = None
                if insurance_img is not None:
                    insurance_ocr = extract_insurance_text(insurance_img)
                    hints = build_hints(insurance_ocr["text"])
                    if api_key():
                        insurance_result = review_insurance(insurance_img, insurance_ocr["text"], hints)

                if id_img is not None:
                    national_result = extract_national_id(id_img)

                st.session_state.v4_routed = routed
                st.session_state.v4_insurance_img = insurance_img
                st.session_state.v4_id_img = id_img
                st.session_state.v4_ocr = insurance_ocr
                st.session_state.v4_result = insurance_result
                st.session_state.v4_national = national_result
                st.session_state.v4_hints = hints or {}
        except Exception as exc:
            st.error(f"فشل التحليل: {exc}")

routed = st.session_state.get("v4_routed", [])
insurance_img = st.session_state.get("v4_insurance_img")
id_img = st.session_state.get("v4_id_img")
insurance_ocr = st.session_state.get("v4_ocr")
insurance_result = st.session_state.get("v4_result")
national_result = st.session_state.get("v4_national")
hints = st.session_state.get("v4_hints", {})

if routed:
    tab1, tab2, tab3, tab4 = st.tabs(["النتيجة", "تصنيف الصور", "OCR الخام", "التدقيق"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("المستند الذي اعتبره كارنيه التأمين")
            if insurance_img is not None:
                st.image(insurance_img, use_container_width=True)
            else:
                st.warning("لم يتم التعرف على كارنيه تأمين")
        with c2:
            st.subheader("المستند الذي اعتبره بطاقة الرقم القومي")
            if id_img is not None:
                st.image(id_img, use_container_width=True)
            else:
                st.warning("لم يتم التعرف على بطاقة رقم قومي")

        direct_nid = national_result.get("value") if national_result else None
        card_id = insurance_result.id_number.value if insurance_result else hints.get("id_number")
        card_nid = valid_egyptian_national_id(card_id)
        final_nid = direct_nid or card_nid

        st.subheader("بيانات المريض")
        st.text_input("الرقم القومي", value=final_nid or "", key="nid_v4")
        if direct_nid:
            st.success(f"الرقم القومي من البطاقة — ثقة OCR {national_result.get('confidence', 0)*100:.0f}%")
        elif card_nid:
            st.info("الرقم القومي مأخوذ من ID No في كارنيه التأمين بعد التحقق البنيوي.")

        if insurance_result:
            st.subheader("بيانات التأمين")
            st.success(insurance_result.summary_ar)
            a, b = st.columns(2)
            a.text_input("الاسم", value=insurance_result.patient_name_en.value or insurance_result.patient_name_ar.value or hints.get("name") or "")
            b.text_input("شركة التأمين", value=insurance_result.insurance_company.value or hints.get("company") or "")
            a.text_input("Card Number", value=insurance_result.card_number.value or hints.get("card_number") or "")
            b.text_input("ID No", value=insurance_result.id_number.value or hints.get("id_number") or "")
            a.text_input("Member ID", value=insurance_result.member_id.value or hints.get("member_id") or "")
            b.text_input("Policy Number", value=insurance_result.policy_number.value or hints.get("policy_number") or "")
            a.text_input("جهة العمل / Policy Holder", value=insurance_result.employer.value or hints.get("employer") or "")
            b.text_input("الفئة / الشبكة", value=insurance_result.network_class.value or hints.get("network_class") or "")
            a.text_input("بداية الصلاحية", value=insurance_result.valid_from.value or hints.get("valid_from") or "")
            b.text_input("تاريخ الانتهاء", value=insurance_result.expiry_date.value or hints.get("expiry_date") or "")
            st.subheader({"valid": "🟢 ساري", "expired": "🔴 منتهي", "unknown": "🟡 غير مؤكد"}[insurance_result.card_status])

    with tab2:
        rows = []
        for item in routed:
            route = item["route"]
            full = route.get("full", {})
            rows.append({
                "الملف": item["name"],
                "التصنيف": route.get("kind"),
                "Insurance score": full.get("insurance_score"),
                "National ID score": full.get("id_score"),
            })
        st.dataframe(rows, use_container_width=True)
        st.caption("لو الصورة تحتوي المستندين، النظام يجرب تقسيمها أعلى/أسفل ويمين/يسار ويختار الجزء الأنسب لكل مستند.")

    with tab3:
        if insurance_ocr:
            st.text_area("OCR كارنيه التأمين", value=insurance_ocr["text"], height=300)
        if national_result:
            st.write("القراءات الخام لمنطقة الرقم القومي")
            st.dataframe(national_result.get("raw_reads", []), use_container_width=True)

    with tab4:
        if hints:
            st.write("الحقول المستخرجة بالقواعد قبل GPT")
            st.json(hints)
        if national_result:
            candidates = national_result.get("candidates", [])
            if candidates:
                st.write("مرشحات الرقم القومي")
                st.dataframe(candidates, use_container_width=True)
        if insurance_result and insurance_result.warnings:
            st.warning("\n".join("• " + x for x in insurance_result.warnings))
