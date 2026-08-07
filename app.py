import base64
import io
import os
from datetime import date
from typing import Literal

import streamlit as st
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel, Field

from src.ocr_engine import extract_insurance_text, extract_national_id


class Field(BaseModel):
    value: str | None = None
    confidence: int = Field(default=0, ge=0, le=100)


class InsuranceResult(BaseModel):
    patient_name_ar: Field
    patient_name_en: Field
    insurance_company: Field
    member_id: Field
    card_number: Field
    policy_number: Field
    employer: Field
    network_class: Field
    valid_from: Field
    expiry_date: Field
    card_status: Literal["valid", "expired", "unknown"]
    warnings: list[str]
    summary_ar: str


INSURANCE_PROMPT = """
أنت مراجع بيانات كارنيه تأمين طبي، وليس OCR.
سيصلك النص الذي استخرجه محرك OCR مستقل، وقد تصلك صورة الكارنيه فقط لفهم مواضع الحقول.

القواعد:
- اعتمد النص المستخرج أولاً.
- لا تخترع أرقاماً غير موجودة في OCR أو واضحة صراحة في الصورة.
- Card Number هو فقط الرقم المرتبط بعنوان Card Number.
- Member ID هو فقط الرقم المرتبط بعنوان ID أو Member ID.
- Policy Number هو فقط الرقم المرتبط بعنوان Policy/Policy No.
- احتفظ بالحروف داخل Card Number إذا كانت مطبوعة ضمنه.
- استخرج جهة العمل والفئة وتواريخ الصلاحية.
- حدّد هل الكارنيه ساري مقارنة بتاريخ اليوم.
- إذا كان الحقل غير واضح أرجع null.
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
st.caption("PaddleOCR يقرأ الأرقام والنص أولاً، وGPT يراجع وينظم فقط.")

with st.sidebar:
    st.info("OCR: PaddleOCR + OpenCV — مجاني")
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
            with st.spinner("تحميل OCR وقراءة المستندات... أول تشغيل قد يستغرق وقتاً لتحميل الموديلات"):
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
            st.info("إذا ظهر خطأ متعلق بـ PaddlePaddle/Python، اضبط Python في Streamlit على 3.13 ثم Reboot.")

insurance_ocr = st.session_state.get("v3_insurance_ocr")
insurance_result = st.session_state.get("v3_insurance_result")
national_result = st.session_state.get("v3_national")

if insurance_ocr or national_result:
    tab1, tab2, tab3, tab4 = st.tabs(["النتيجة", "OCR الخام", "معالجة الصور", "التدقيق"])

    with tab1:
        if national_result:
            st.subheader("بطاقة الرقم القومي")
            nid = national_result.get("value")
            conf = national_result.get("confidence", 0.0)
            st.text_input("الرقم القومي", value=nid or "", key="nid_v3_result")
            if nid:
                st.success(f"تمت قراءة 14 رقم — ثقة OCR {conf*100:.0f}%")
            else:
                st.warning("لم يعتمد OCR رقماً قومياً بعد. راجع تبويب التدقيق لمعرفة القراءات المرشحة.")

        if insurance_result:
            st.subheader("بيانات التأمين")
            st.success(insurance_result.summary_ar)
            a, b = st.columns(2)
            a.text_input("الاسم", value=insurance_result.patient_name_en.value or insurance_result.patient_name_ar.value or "")
            b.text_input("شركة التأمين", value=insurance_result.insurance_company.value or "")
            a.text_input("Card Number", value=insurance_result.card_number.value or "")
            b.text_input("Member ID", value=insurance_result.member_id.value or "")
            a.text_input("Policy Number", value=insurance_result.policy_number.value or "")
            b.text_input("جهة العمل", value=insurance_result.employer.value or "")
            a.text_input("الفئة / الشبكة", value=insurance_result.network_class.value or "")
            b.text_input("تاريخ الانتهاء", value=insurance_result.expiry_date.value or "")
            status = {"valid":"🟢 ساري", "expired":"🔴 منتهي", "unknown":"🟡 غير مؤكد"}[insurance_result.card_status]
            st.subheader(status)
        elif insurance_ocr:
            st.info("تم OCR للكارنيه. أضف API Key ليقوم GPT بتنظيم الحقول، أو راجع النص الخام.")

    with tab2:
        if insurance_ocr:
            st.text_area("نص PaddleOCR", value=insurance_ocr["text"], height=350)
            st.dataframe(insurance_ocr["lines"], use_container_width=True)

    with tab3:
        cols = st.columns(2)
        if insurance_ocr:
            cols[0].image(insurance_ocr["enhanced"], caption="كارنيه محسن بـ OpenCV", use_container_width=True)
        if national_result and national_result.get("roi") is not None:
            cols[1].image(national_result["roi"], caption="منطقة الرقم القومي التي قرأها OCR", use_container_width=True)

    with tab4:
        if national_result:
            st.write("مرشحات الرقم القومي من قراءات مستقلة:")
            candidates = [{k:v for k,v in item.items() if k != "roi"} for item in national_result.get("candidates", [])]
            if candidates:
                st.dataframe(candidates, use_container_width=True)
            else:
                st.warning("لم يجد PaddleOCR مرشحاً صالحاً من 14 رقم في منطقة الرقم.")
            if national_result.get("agreement"):
                st.metric("عدد القراءات المتفقة", national_result["agreement"])

        if insurance_result and insurance_result.warnings:
            st.warning("\n".join("• " + x for x in insurance_result.warnings))
