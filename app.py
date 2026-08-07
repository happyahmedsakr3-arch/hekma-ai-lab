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

from src.insurance_parser import build_hints
from src.ocr_engine import extract_national_id
from src.text_router import route_image_by_text


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
وصلك نص OCR من الجزء الذي اختاره محرك Text-First باعتباره كارنيه التأمين، ومعه HINTS مبنية من العناوين المطبوعة.

قواعد إلزامية:
- لا تخلط بيانات بطاقة الرقم القومي مع بيانات كارنيه التأمين.
- Card Number = فقط القيمة بجوار Card Number / Card No.
- ID No / ID Number = حصراً في id_number.
- Member ID = فقط إذا كان العنوان Member ID / Membership ID.
- Policy Number = فقط إذا كان العنوان Policy No / Policy Number.
- Policy Holder = جهة العمل/صاحب الوثيقة، وليس Policy Number.
- احتفظ بالحروف داخل Card Number كما هي.
- لا تخترع أي رقم. إذا لم يوجد الحقل صراحة أرجع null.
- HINTS المأخوذة من label واضح لها أولوية، واستخدم الصورة فقط لحسم اللبس.
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


st.set_page_config(page_title="Hekma AI V5", page_icon="🧠", layout="wide")
st.markdown("""
<style>
html,body,[class*='css']{direction:rtl;text-align:right}
.block-container{max-width:1180px;padding-top:2rem}
</style>
""", unsafe_allow_html=True)

st.title("🧠 Hekma AI — Document AI V5")
st.caption("Text-first: نقرأ النص أولاً، ثم نحدد نوع المستند من محتواه — الترتيب غير مهم.")

with st.sidebar:
    st.info("OCR مجاني: Tesseract عربي/إنجليزي")
    st.success("GPT متصل") if api_key() else st.warning("GPT غير متصل")

files = st.file_uploader(
    "ارفع صورة أو عدة صور — ممكن بطاقة، كارنيه، أو صورة فيها الاتنين",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True,
    key="documents_v5",
)

if files:
    cols = st.columns(min(3, len(files)))
    for i, file in enumerate(files):
        cols[i % len(cols)].image(file, caption=file.name, use_container_width=True)

if st.button("تحليل V5", type="primary", use_container_width=True):
    if not files:
        st.error("ارفع صورة واحدة على الأقل")
    else:
        try:
            with st.spinner("جاري OCR شامل لكل صورة ثم تحديد نوع المستند من النص..."):
                routed = []
                insurance_candidates = []
                id_candidates = []

                for file in files:
                    image = open_upload(file)
                    route = route_image_by_text(image)
                    routed.append({"name": file.name, "route": route})

                    if route.get("insurance_image") is not None:
                        insurance_candidates.append({
                            "name": file.name,
                            "image": route["insurance_image"],
                            "ocr": route.get("insurance_ocr"),
                            "score": route.get("insurance_score", 0),
                            "segment": route.get("insurance_segment"),
                        })

                    if route.get("id_image") is not None:
                        id_candidates.append({
                            "name": file.name,
                            "image": route["id_image"],
                            "score": route.get("id_score", 0),
                            "segment": route.get("id_segment"),
                        })

                best_ins = max(insurance_candidates, key=lambda x: x["score"]) if insurance_candidates else None
                best_id = max(id_candidates, key=lambda x: x["score"]) if id_candidates else None

                insurance_img = best_ins["image"] if best_ins else None
                id_img = best_id["image"] if best_id else None
                insurance_ocr = best_ins["ocr"] if best_ins else None

                hints = insurance_result = national_result = None

                if insurance_img is not None and insurance_ocr:
                    hints = build_hints(insurance_ocr.get("text", ""))
                    if api_key():
                        insurance_result = review_insurance(insurance_img, insurance_ocr.get("text", ""), hints)

                if id_img is not None:
                    national_result = extract_national_id(id_img)

                st.session_state.v5_routed = routed
                st.session_state.v5_best_ins = best_ins
                st.session_state.v5_best_id = best_id
                st.session_state.v5_ocr = insurance_ocr
                st.session_state.v5_result = insurance_result
                st.session_state.v5_national = national_result
                st.session_state.v5_hints = hints or {}
        except Exception as exc:
            st.error(f"فشل التحليل: {exc}")

routed = st.session_state.get("v5_routed", [])
best_ins = st.session_state.get("v5_best_ins")
best_id = st.session_state.get("v5_best_id")
insurance_ocr = st.session_state.get("v5_ocr")
insurance_result = st.session_state.get("v5_result")
national_result = st.session_state.get("v5_national")
hints = st.session_state.get("v5_hints", {})

if routed:
    tab1, tab2, tab3, tab4 = st.tabs(["النتيجة", "فهم المستندات", "OCR الخام", "التدقيق"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("كارنيه التأمين المختار")
            if best_ins:
                st.caption(f"{best_ins['name']} — الجزء: {best_ins['segment']} — Score: {best_ins['score']}")
                st.image(best_ins["image"], use_container_width=True)
            else:
                st.warning("لم يجد النص مؤشرات كافية لكارنيه تأمين")
        with c2:
            st.subheader("بطاقة الرقم القومي المختارة")
            if best_id:
                st.caption(f"{best_id['name']} — الجزء: {best_id['segment']} — Score: {best_id['score']}")
                st.image(best_id["image"], use_container_width=True)
            else:
                st.warning("لم يجد النص مؤشرات كافية لبطاقة رقم قومي")

        direct_nid = national_result.get("value") if national_result else None
        card_id = insurance_result.id_number.value if insurance_result else hints.get("id_number")
        card_nid = valid_egyptian_national_id(card_id)
        final_nid = direct_nid or card_nid

        st.subheader("بيانات المريض")
        st.text_input("الرقم القومي", value=final_nid or "", key="nid_v5")
        if direct_nid:
            st.success(f"الرقم القومي من البطاقة — ثقة OCR {national_result.get('confidence', 0)*100:.0f}%")
        elif card_nid:
            st.info("تم أخذ الرقم القومي من ID No في كارنيه التأمين بعد التحقق البنيوي.")
        elif best_id:
            st.warning("تم التعرف على البطاقة، لكن رقمها لم ينجح في التحقق بعد.")

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
        elif insurance_ocr:
            st.subheader("بيانات التأمين من القواعد المباشرة")
            st.json(hints)

    with tab2:
        rows = []
        for item in routed:
            route = item["route"]
            for segment in route.get("segments", []):
                rows.append({
                    "الملف": item["name"],
                    "الجزء": segment.get("label"),
                    "نوع النص": segment.get("kind"),
                    "Insurance score": segment.get("insurance_score"),
                    "National ID score": segment.get("id_score"),
                    "Insurance hits": ", ".join(segment.get("insurance_hits", [])),
                    "ID hits": ", ".join(segment.get("id_hits", [])),
                })
        st.dataframe(rows, use_container_width=True)
        st.caption("V5 لا يفترض نوع الصورة. يعمل OCR على الصورة والأجزاء المحتملة، ثم يختار حسب محتوى النص.")

    with tab3:
        if insurance_ocr:
            st.text_area("OCR الجزء المختار ككارنيه تأمين", value=insurance_ocr.get("text", ""), height=320)
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
