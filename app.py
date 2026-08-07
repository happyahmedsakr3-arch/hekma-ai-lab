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
سيصلك النص الذي استخرجه Tesseract OCR وصورة الكارنيه لفهم مواضع الحقول.

قواعد إلزامية:
- اعتمد النص المستخرج أولاً واستخدم الصورة فقط لحسم عنوان الحقل.
- لا تخترع أرقاماً غير موجودة في OCR أو واضحة صراحة في الصورة.
- Card Number = فقط القيمة بجوار Card Number.
- ID No أو ID Number = يوضع حصراً في id_number. لا تضعه في Member ID.
- Member ID = فقط إذا كان العنوان المطبوع نفسه Member ID أو Membership ID. غير ذلك null.
- Policy Number = فقط إذا كان العنوان Policy / Policy No / Policy Number. غير ذلك null.
- لا تنقل Policy Holder إلى Policy Number؛ Policy Holder غالباً جهة العمل/صاحب الوثيقة.
- احتفظ بالحروف داخل Card Number إذا كانت مطبوعة ضمن الرقم.
- استخرج جهة العمل والفئة وتواريخ الصلاحية.
- حدّد هل الكارنيه ساري مقارنة بتاريخ اليوم.
- إذا كان الحقل غير واضح أرجع null.
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


def review_insurance(image: Image.Image, ocr_text: str) -> InsuranceResult:
    if not api_key():
        raise RuntimeError("OpenAI API Key غير موجود")
    client = OpenAI(api_key=api_key())
    content = [
        {"type": "input_text", "text": f"تاريخ اليوم: {date.today().isoformat()}\n\nOCR TEXT:\n{ocr_text}"},
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


st.set_page_config(page_title="Hekma AI V3", page_icon="🧠", layout="wide")
st.markdown(
    """
    <style>
    html,body,[class*='css']{direction:rtl;text-align:right}
    .block-container{max-width:1180px;padding-top:2rem}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧠 Hekma AI — OCR V3")
st.caption("Tesseract OCR يقرأ النص والأرقام أولاً، وGPT يراجع توزيع الحقول فقط.")

with st.sidebar:
    st.info("OCR: Tesseract عربي/إنجليزي — مجاني")
    st.success("GPT متصل") if api_key() else st.warning("GPT غير متصل — OCR يظل قابلاً للاختبار")

c1, c2 = st.columns(2)
with c1:
    insurance_file = st.file_uploader("كارنيه التأمين", type=["png", "jpg", "jpeg", "webp"], key="insurance_v3")
with c2:
    id_file = st.file_uploader("بطاقة الرقم القومي", type=["png", "jpg", "jpeg", "webp"], key="id_v3")

if insurance_file or id_file:
    p1, p2 = st.columns(2)
    if insurance_file:
        p1.image(insurance_file, caption="كارنيه التأمين", use_container_width=True)
    if id_file:
        p2.image(id_file, caption="بطاقة الرقم القومي", use_container_width=True)

if st.button("تشغيل OCR V3", type="primary", use_container_width=True):
    if not insurance_file and not id_file:
        st.error("ارفع مستنداً واحداً على الأقل")
    else:
        try:
            with st.spinner("جاري قراءة المستندات وتدقيق الحقول..."):
                insurance_ocr = None
                insurance_result = None
                national_result = None

                if insurance_file:
                    ins_img = open_upload(insurance_file)
                    insurance_ocr = extract_insurance_text(ins_img)
                    if api_key():
                        insurance_result = review_insurance(ins_img, insurance_ocr["text"])

                if id_file:
                    id_img = open_upload(id_file)
                    national_result = extract_national_id(id_img)

                st.session_state.v3_insurance_ocr = insurance_ocr
                st.session_state.v3_insurance_result = insurance_result
                st.session_state.v3_national = national_result
        except Exception as exc:
            st.error(f"فشل OCR V3: {exc}")

insurance_ocr = st.session_state.get("v3_insurance_ocr")
insurance_result = st.session_state.get("v3_insurance_result")
national_result = st.session_state.get("v3_national")

if insurance_ocr or national_result:
    tab1, tab2, tab3, tab4 = st.tabs(["النتيجة", "OCR الخام", "معالجة الصور", "التدقيق"])

    with tab1:
        direct_nid = national_result.get("value") if national_result else None
        card_nid = valid_egyptian_national_id(insurance_result.id_number.value) if insurance_result else None
        final_nid = direct_nid or card_nid

        st.subheader("بيانات رقم الهوية")
        st.text_input("الرقم القومي", value=final_nid or "", key="nid_v3_result")
        if direct_nid:
            conf = national_result.get("confidence", 0.0)
            st.success(f"تمت القراءة من بطاقة الرقم القومي — ثقة OCR {conf*100:.0f}%")
        elif card_nid:
            st.info("تم أخذ الرقم القومي من خانة ID No في كارنيه التأمين بعد التحقق من أنه 14 رقماً وبنية تاريخ صحيحة.")
        elif id_file:
            st.warning("لم يعتمد OCR الرقم القومي من البطاقة حتى الآن. راجع تبويب التدقيق.")

        if insurance_result:
            st.subheader("بيانات التأمين")
            st.success(insurance_result.summary_ar)
            a, b = st.columns(2)
            a.text_input("الاسم", value=insurance_result.patient_name_en.value or insurance_result.patient_name_ar.value or "")
            b.text_input("شركة التأمين", value=insurance_result.insurance_company.value or "")
            a.text_input("Card Number", value=insurance_result.card_number.value or "")
            b.text_input("ID No", value=insurance_result.id_number.value or "")
            a.text_input("Member ID", value=insurance_result.member_id.value or "")
            b.text_input("Policy Number", value=insurance_result.policy_number.value or "")
            a.text_input("جهة العمل / Policy Holder", value=insurance_result.employer.value or "")
            b.text_input("الفئة / الشبكة", value=insurance_result.network_class.value or "")
            a.text_input("بداية الصلاحية", value=insurance_result.valid_from.value or "")
            b.text_input("تاريخ الانتهاء", value=insurance_result.expiry_date.value or "")
            status = {"valid": "🟢 ساري", "expired": "🔴 منتهي", "unknown": "🟡 غير مؤكد"}[insurance_result.card_status]
            st.subheader(status)
        elif insurance_ocr:
            st.info("تم OCR للكارنيه. أضف API Key ليقوم GPT بتنظيم الحقول، أو راجع النص الخام.")

    with tab2:
        if insurance_ocr:
            st.text_area("نص Tesseract OCR", value=insurance_ocr["text"], height=350)
            st.dataframe(insurance_ocr["lines"], use_container_width=True)

    with tab3:
        cols = st.columns(2)
        if insurance_ocr:
            cols[0].image(insurance_ocr["enhanced"], caption="كارنيه محسن قبل OCR", use_container_width=True)
        if national_result and national_result.get("roi") is not None:
            cols[1].image(national_result["roi"], caption="منطقة الرقم القومي التي قرأها OCR", use_container_width=True)

    with tab4:
        if national_result:
            st.write("مرشحات الرقم القومي من قراءات Tesseract مستقلة:")
            candidates = [{k: v for k, v in item.items() if k != "roi"} for item in national_result.get("candidates", [])]
            if candidates:
                st.dataframe(candidates, use_container_width=True)
            else:
                st.warning("لم يجد Tesseract مرشحاً صالحاً من 14 رقم في منطقة الرقم.")
            raw_reads = national_result.get("raw_reads", [])
            if raw_reads:
                st.write("القراءات الخام لمنطقة الرقم القومي:")
                st.dataframe(raw_reads, use_container_width=True)
            if national_result.get("agreement"):
                st.metric("عدد القراءات المتفقة", national_result["agreement"])

        if insurance_result and insurance_result.warnings:
            st.warning("\n".join("• " + x for x in insurance_result.warnings))
