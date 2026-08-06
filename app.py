import base64
import io
import os
import re
from datetime import date
from typing import Literal

import streamlit as st
from openai import OpenAI
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from pydantic import BaseModel, Field


class ExtractedField(BaseModel):
    value: str | None = None
    confidence: int = Field(default=0, ge=0, le=100)
    source: str | None = None


class ReaderResult(BaseModel):
    patient_name_ar: ExtractedField
    patient_name_en: ExtractedField
    national_id: ExtractedField
    insurance_company: ExtractedField
    member_id: ExtractedField
    card_number: ExtractedField
    policy_number: ExtractedField
    employer: ExtractedField
    network_class: ExtractedField
    expiry_date: ExtractedField
    card_status: Literal["valid", "expired", "unknown"]
    names_match: Literal["match", "possible_match", "mismatch", "unknown"]
    warnings: list[str]
    summary_ar: str


PROMPT = """
أنت محرك تدقيق مستندات طبية. ستستقبل صورًا أصلية وصورًا محسنة وقصاصة مخصصة لسطر الرقم القومي.

قواعد إلزامية:
1. استخدم الصورة الأصلية لفهم نوع المستند والأسماء والشركة.
2. استخدم الصورة المحسنة للأرقام المطبوعة على كارنيه التأمين.
3. استخدم قصاصة الرقم القومي فقط لقراءة الرقم القومي.
4. الرقم القومي لا يُقبل إلا إذا كان 14 رقمًا واضحًا. عند أي شك أرجع null.
5. لا تخمن أي رقم، ولا تكمل أرقامًا ناقصة من السياق.
6. لا تعتبر الكود اللاتيني أسفل البطاقة رقمًا قوميًا.
7. Member ID هو الرقم بجوار ID أو Member ID فقط.
8. Card Number هو الرقم بجوار Card Number فقط. إن لم يوجد عنوان واضح أرجع null.
9. فرّق بين Policy No وCard Number وMember ID.
10. حدّد صلاحية الكارنيه من تاريخ الانتهاء مقارنة بتاريخ اليوم.
11. أرجع نتيجة منظمة فقط.
"""

ARABIC_TO_LATIN = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def api_key() -> str:
    try:
        return st.secrets.get("OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
    except Exception:
        return os.getenv("OPENAI_API_KEY", "")


def image_to_data_url(image: Image.Image, quality: int = 95) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def open_uploaded(uploaded_file) -> Image.Image:
    return Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")


def enhance_document(image: Image.Image) -> Image.Image:
    max_side = 2200
    scale = min(max_side / max(image.size), 1.0)
    if scale < 1.0:
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)

    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    gray = ImageEnhance.Contrast(gray).enhance(1.8)
    gray = ImageEnhance.Sharpness(gray).enhance(2.2)
    gray = gray.filter(ImageFilter.UnsharpMask(radius=1.8, percent=170, threshold=3))
    return gray.convert("RGB")


def national_id_crop(image: Image.Image) -> Image.Image:
    # البطاقة المصرية يكون الرقم القومي غالبًا في الشريط السفلي يمين البطاقة.
    w, h = image.size
    crop = image.crop((int(w * 0.32), int(h * 0.63), int(w * 0.99), int(h * 0.94)))
    crop = crop.resize((crop.width * 4, crop.height * 4), Image.Resampling.LANCZOS)
    crop = ImageOps.grayscale(crop)
    crop = ImageOps.autocontrast(crop, cutoff=1)
    crop = ImageEnhance.Contrast(crop).enhance(2.4)
    crop = ImageEnhance.Sharpness(crop).enhance(3.0)
    crop = crop.filter(ImageFilter.UnsharpMask(radius=2, percent=220, threshold=2))
    return crop.convert("RGB")


def normalized_digits(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = re.sub(r"\D", "", value.translate(ARABIC_TO_LATIN))
    return cleaned or None


def strict_national_id(value: str | None) -> str | None:
    cleaned = normalized_digits(value)
    if not cleaned or len(cleaned) != 14 or cleaned[0] not in {"2", "3"}:
        return None
    try:
        century = 1900 if cleaned[0] == "2" else 2000
        date(century + int(cleaned[1:3]), int(cleaned[3:5]), int(cleaned[5:7]))
    except ValueError:
        return None
    return cleaned


def analyze_documents(key: str, insurance_file, id_file) -> tuple[ReaderResult, dict[str, Image.Image]]:
    insurance_original = open_uploaded(insurance_file) if insurance_file else None
    id_original = open_uploaded(id_file) if id_file else None

    prepared: dict[str, Image.Image] = {}
    content: list[dict] = [{
        "type": "input_text",
        "text": f"حلل المستندات التالية. تاريخ اليوم {date.today().isoformat()}. كل صورة مسبوقة بوصف يحدد وظيفتها.",
    }]

    if insurance_original:
        insurance_enhanced = enhance_document(insurance_original)
        prepared["insurance_enhanced"] = insurance_enhanced
        content.extend([
            {"type": "input_text", "text": "صورة كارنيه التأمين الأصلية:"},
            {"type": "input_image", "image_url": image_to_data_url(insurance_original), "detail": "high"},
            {"type": "input_text", "text": "نسخة محسنة من كارنيه التأمين لقراءة الأرقام والعناوين:"},
            {"type": "input_image", "image_url": image_to_data_url(insurance_enhanced), "detail": "high"},
        ])

    if id_original:
        id_enhanced = enhance_document(id_original)
        id_number_crop = national_id_crop(id_original)
        prepared["id_enhanced"] = id_enhanced
        prepared["id_number_crop"] = id_number_crop
        content.extend([
            {"type": "input_text", "text": "صورة بطاقة الرقم القومي الأصلية:"},
            {"type": "input_image", "image_url": image_to_data_url(id_original), "detail": "high"},
            {"type": "input_text", "text": "نسخة محسنة من بطاقة الرقم القومي:"},
            {"type": "input_image", "image_url": image_to_data_url(id_enhanced), "detail": "high"},
            {"type": "input_text", "text": "قصاصة مخصصة لسطر الرقم القومي فقط. اقرأ منها 14 رقمًا أو أرجع null:"},
            {"type": "input_image", "image_url": image_to_data_url(id_number_crop), "detail": "high"},
        ])

    client = OpenAI(api_key=key)
    response = client.responses.parse(
        model=os.getenv("OPENAI_MODEL", "gpt-5.1"),
        instructions=PROMPT,
        input=[{"role": "user", "content": content}],
        text_format=ReaderResult,
    )
    if response.output_parsed is None:
        raise RuntimeError("لم يتم استخراج نتيجة منظمة")
    return response.output_parsed, prepared


st.set_page_config(page_title="Hekma AI V2", page_icon="🧠", layout="wide")
st.markdown(
    "<style>html,body,[class*='css']{direction:rtl;text-align:right}.block-container{max-width:1150px;padding-top:2rem}</style>",
    unsafe_allow_html=True,
)

st.title("🧠 Hekma AI — Vision Pipeline V2")
st.caption("ارفع الكارنيه والبطاقة في خانات منفصلة. النظام يحسن الصور ويقص منطقة الرقم القومي تلقائيًا.")

with st.sidebar:
    st.success("API متصل") if api_key() else st.error("API غير متصل")

col1, col2 = st.columns(2)
with col1:
    insurance_file = st.file_uploader("كارنيه التأمين", type=["png", "jpg", "jpeg", "webp"], key="insurance")
with col2:
    id_file = st.file_uploader("بطاقة الرقم القومي", type=["png", "jpg", "jpeg", "webp"], key="national_id")

if insurance_file or id_file:
    p1, p2 = st.columns(2)
    if insurance_file:
        with p1:
            st.image(insurance_file, caption="كارنيه التأمين الأصلي", use_container_width=True)
    if id_file:
        with p2:
            st.image(id_file, caption="بطاقة الرقم القومي الأصلية", use_container_width=True)

if st.button("تحليل V2", type="primary", use_container_width=True):
    if not api_key():
        st.error("مفتاح OpenAI غير موجود")
    elif not insurance_file and not id_file:
        st.error("ارفع مستندًا واحدًا على الأقل")
    else:
        try:
            with st.spinner("جاري تحسين الصور، قص مناطق الأرقام، ثم التحليل..."):
                result, prepared = analyze_documents(api_key(), insurance_file, id_file)
                result.national_id.value = strict_national_id(result.national_id.value)
                if not result.national_id.value:
                    result.national_id.confidence = 0
                    result.warnings.append("الرقم القومي لم ينجح في التحقق الصارم؛ تم تركه فارغًا بدل عرض رقم خاطئ")
                st.session_state.result_v2 = result
                st.session_state.prepared_v2 = prepared
        except Exception as exc:
            st.error(f"فشل التحليل: {exc}")

result = st.session_state.get("result_v2")
prepared = st.session_state.get("prepared_v2", {})

if result:
    st.success(result.summary_ar)
    st.subheader({"valid": "🟢 الكارنيه ساري", "expired": "🔴 الكارنيه منتهي", "unknown": "🟡 الصلاحية غير واضحة"}[result.card_status])

    patient_tab, insurance_tab, processing_tab, json_tab = st.tabs(["بيانات المريض", "بيانات التأمين", "معالجة الصور", "JSON"])

    with patient_tab:
        c1, c2 = st.columns(2)
        c1.text_input("الاسم بالعربي", value=result.patient_name_ar.value or "")
        c2.text_input("الاسم بالإنجليزي", value=result.patient_name_en.value or "")
        st.text_input("الرقم القومي المؤكد", value=result.national_id.value or "")
        if not result.national_id.value:
            st.error("الرقم القومي غير مؤكد. ارفع صورة أقرب للبطاقة بدل اعتماد رقم خاطئ.")

    with insurance_tab:
        c1, c2 = st.columns(2)
        c1.text_input("شركة التأمين", value=result.insurance_company.value or "")
        c2.text_input("جهة العمل", value=result.employer.value or "")
        c1.text_input("Member ID", value=result.member_id.value or "")
        c2.text_input("Card Number", value=result.card_number.value or "")
        c1.text_input("Policy Number", value=result.policy_number.value or "")
        c2.text_input("الفئة / الشبكة", value=result.network_class.value or "")
        st.text_input("تاريخ الانتهاء", value=result.expiry_date.value or "")

    with processing_tab:
        if prepared:
            cols = st.columns(min(len(prepared), 3))
            for index, (name, image) in enumerate(prepared.items()):
                with cols[index % len(cols)]:
                    st.image(image, caption=name, use_container_width=True)
        else:
            st.info("لا توجد صور معالجة بعد")

    with json_tab:
        st.json(result.model_dump())

    if result.warnings:
        st.warning("\n".join(f"• {item}" for item in dict.fromkeys(result.warnings)))
