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

from src.ocr_engine import extract_insurance_text, extract_national_id
from src.insurance_parser import build_hints


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
سيصلك نص Tesseract وصورة الكارنيه، ومعهما HINTS مستخرجة بقواعد ثابتة من أسماء الحقول المطبوعة.

الأولوية:
1) إذا كانت HINTS تحتوي قيمة واضحة مرتبطة بعنوان مطبوع، حافظ عليها ولا تنقلها لحقل آخر.
2) استخدم الصورة فقط للمراجعة وحسم اللبس.
3) لا تخترع أي رقم.

قواعد الحقول:
- Card Number = فقط القيمة بجوار Card Number / Card No.
- ID No / ID Number = حصراً في id_number.
- Member ID = فقط إذا كان العنوان Member ID / Membership ID.
- Policy Number = فقط إذا كان العنوان Policy No / Policy Number.
- Policy Holder = جهة العمل/صاحب الوثيقة وليس Policy Number.
- احتفظ بالحروف داخل Card Number.
- إذا لم يوجد الحقل صراحة أرجع null.
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
        {
            "type": "input_text",
            "text": f"تاريخ اليوم: {date.today().isoformat()}\n\nHINTS:\n{hints}\n\nOCR TEXT:\n{ocr_text}",
        },
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


st.set_page_config(page_title="Hekma AI V3.2", page_icon="🧠", layout="wide")
st.markdown("""
<style>
html,body,[class*='css']{direction:rtl;text-align:right}
.block-container{max-width:1180px;padding-top:2rem}
</style>
""", unsafe_allow_html=True)

st.title("🧠 Hekma AI — OCR V3.2")
st.caption("Tesseract يقرأ أولاً، قواعد ثابتة تفهم عناوين الحقول، وGPT يراجع فقط.")

with st.sidebar:
    st.info("OCR: Tesseract عربي/إنجليزي — مجاني")
    st.success("GPT متصل") if api_key() else st.warning("GPT غير متصل")

c1, c2 = st.columns(2)
with c1:
    insurance_file = st.file_uploader("كارنيه التأمين", type=["png", "jpg", "jpeg", "webp"], key="insurance_v32")
with c2:
    id_file = st.file_uploader("بطاقة الرقم القومي", type=["png", "jpg", "jpeg", "webp"], key="id_v32")

if insurance_file or id_file:
    p1, p2 = st.columns(2)
    if insurance_file:
        p1.image(insurance_file, caption="كارنيه التأمين", use_container_width=True)
    if id_file:
        p2.image(id_file, caption="بطاقة الرقم القومي", use_container_width=True)

if st.button("تشغيل OCR V3.2", type="primary", use_container_width=True):
    if not insurance_file and not id_file:
        st.error("ارفع مستنداً واحداً على الأقل")
    else:
        try:
            with st.spinner("جاري OCR ثم مطابقة عناوين الحقول والتدقيق..."):
                insurance_ocr = insurance_result = national_result = hints = None

                if insurance_file:
                    ins_img = open_upload(insurance_file)
                    insurance_ocr = extract_insurance_text(ins_img)
                    hints = build_hints(insurance_ocr["text"])
                    if api_key():
                        insurance_result = review_insurance(ins_img, insurance_ocr["text"], hints)

                if id_file:
                    national_result = extract_national_id(open_upload(id_file))

                st.session_state.v32_ocr = insurance_ocr
                st.session_state.v32_result = insurance_result
                st.session_state.v32_national = national_result
                st.session_state.v32_hints = hints
        except Exception as exc:
            st.error(f"فشل OCR V3.2: {exc}")

insurance_ocr = st.session_state.get("v32_ocr")
insurance_result = st.session_state.get("v32_result")
national_result = st.session_state.get("v32_national")
hints = st.session_state.get("v32_hints") or {}

if insurance_ocr or national_result:
    tab1, tab2, tab3, tab4 = st.tabs(["النتيجة", "OCR الخام", "معالجة الصور", "التدقيق"])

    with tab1:
        direct_nid = national_result.get("value") if national_result else None
        card_id = insurance_result.id_number.value if insurance_result else hints.get("id_number")
        card_nid = valid_egyptian_national_id(card_id)
        final_nid = direct_nid or card_nid

        st.subheader("بيانات رقم الهوية")
        st.text_input("الرقم القومي", value=final_nid or "", key="nid_v32_result")
        if direct_nid:
            st.success(f"الرقم القومي من البطاقة — ثقة OCR {national_result.get('confidence', 0)*100:.0f}%")
        elif card_nid:
            st.info("تم أخذ الرقم القومي من ID No في كارنيه التأمين بعد التحقق البنيوي.")
        elif id_file:
            st.warning("لم يعتمد OCR الرقم القومي بعد؛ راجع التدقيق الخام.")

        detected_company = hints.get("company")
        if detected_company:
            st.caption(f"Template detected: {detected_company}")

        if insurance_result:
            st.subheader("بيانات التأمين")
            st.success(insurance_result.summary_ar)
            a, b = st.columns(2)
            a.text_input("الاسم", value=insurance_result.patient_name_en.value or insurance_result.patient_name_ar.value or hints.get("name") or "")
            b.text_input("شركة التأمين", value=insurance_result.insurance_company.value or detected_company or "")
            a.text_input("Card Number", value=insurance_result.card_number.value or hints.get("card_number") or "")
            b.text_input("ID No", value=insurance_result.id_number.value or hints.get("id_number") or "")
            a.text_input("Member ID", value=insurance_result.member_id.value or hints.get("member_id") or "")
            b.text_input("Policy Number", value=insurance_result.policy_number.value or hints.get("policy_number") or "")
            a.text_input("جهة العمل / Policy Holder", value=insurance_result.employer.value or hints.get("employer") or "")
            b.text_input("الفئة / الشبكة", value=insurance_result.network_class.value or hints.get("network_class") or "")
            a.text_input("بداية الصلاحية", value=insurance_result.valid_from.value or hints.get("valid_from") or "")
            b.text_input("تاريخ الانتهاء", value=insurance_result.expiry_date.value or hints.get("expiry_date") or "")
            status = {"valid": "🟢 ساري", "expired": "🔴 منتهي", "unknown": "🟡 غير مؤكد"}[insurance_result.card_status]
            st.subheader(status)
        elif insurance_ocr:
            st.subheader("الحقول من القواعد المباشرة")
            st.json(hints)

    with tab2:
        if insurance_ocr:
            st.text_area("نص Tesseract OCR", value=insurance_ocr["text"], height=350)

    with tab3:
        cols = st.columns(2)
        if insurance_ocr:
            cols[0].image(insurance_ocr["enhanced"], caption="كارنيه محسن قبل OCR", use_container_width=True)
        if national_result and national_result.get("roi") is not None:
            cols[1].image(national_result["roi"], caption="منطقة الرقم القومي", use_container_width=True)

    with tab4:
        if hints:
            st.write("الحقول التي استخرجتها القواعد قبل GPT:")
            st.json(hints)
        if national_result:
            candidates = national_result.get("candidates", [])
            raw_reads = national_result.get("raw_reads", [])
            if candidates:
                st.write("مرشحات الرقم القومي:")
                st.dataframe(candidates, use_container_width=True)
            if raw_reads:
                st.write("القراءات الخام للرقم القومي:")
                st.dataframe(raw_reads, use_container_width=True)
        if insurance_result and insurance_result.warnings:
            st.warning("\n".join("• " + x for x in insurance_result.warnings))
