import base64
import io
import os
import re
from collections import Counter
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


class NationalIdReads(BaseModel):
    original_read: str | None = None
    grayscale_read: str | None = None
    threshold_read: str | None = None
    confidence: int = Field(default=0, ge=0, le=100)
    note: str | None = None


PROMPT = """
أنت محرك تدقيق مستندات طبية.

قواعد إلزامية:
1. استخدم الصورة الأصلية لفهم نوع المستند والأسماء والشركة.
2. استخدم الصورة المحسنة للأرقام المطبوعة على كارنيه التأمين.
3. لا تعتمد الرقم القومي من التحليل العام؛ سيُقرأ في مرحلة مستقلة.
4. لا تخمن أي رقم، ولا تكمل أرقامًا ناقصة من السياق.
5. Member ID هو الرقم بجوار ID أو Member ID فقط.
6. Card Number هو الرقم بجوار Card Number فقط. إن لم يوجد عنوان واضح أرجع null.
7. فرّق بين Policy No وCard Number وMember ID.
8. حدّد صلاحية الكارنيه من تاريخ الانتهاء مقارنة بتاريخ اليوم.
9. أرجع نتيجة منظمة فقط.
"""

NATIONAL_ID_PROMPT = """
أنت آلة نسخ أرقام فقط، وليست مهمتك تفسير الصورة.
سترى ثلاث نسخ لنفس سطر الرقم القومي المصري: أصلية، رمادية عالية التباين، وأبيض/أسود.

لكل نسخة:
- انسخ 14 رقمًا كما تظهر بصريًا فقط.
- حوّل الأرقام العربية المطبوعة إلى أرقام 0-9 في الناتج.
- لا تستخدم الاسم أو تاريخ الميلاد أو أي سياق لتخمين رقم.
- تجاهل الكود اللاتيني أسفل البطاقة.
- إذا لم تستطع تمييز 14 خانة كاملة في نسخة ما، أرجع null لهذه النسخة.
- لا تُصلح رقمًا ولا تستنتج رقمًا ناقصًا.
"""

ARABIC_TO_LATIN = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def api_key() -> str:
    try:
        return st.secrets.get("OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
    except Exception:
        return os.getenv("OPENAI_API_KEY", "")


def image_to_data_url(image: Image.Image, quality: int = 96) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def open_uploaded(uploaded_file) -> Image.Image:
    return Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")


def enhance_document(image: Image.Image) -> Image.Image:
    max_side = 2400
    scale = min(max_side / max(image.size), 1.0)
    if scale < 1.0:
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    gray = ImageEnhance.Contrast(gray).enhance(1.7)
    gray = ImageEnhance.Sharpness(gray).enhance(2.0)
    return gray.filter(ImageFilter.UnsharpMask(radius=1.5, percent=160, threshold=3)).convert("RGB")


def national_id_crops(image: Image.Image) -> dict[str, Image.Image]:
    w, h = image.size
    base = image.crop((int(w * 0.30), int(h * 0.64), int(w * 0.995), int(h * 0.93)))
    base = base.resize((max(base.width * 6, 1800), max(base.height * 6, 360)), Image.Resampling.LANCZOS)

    original = ImageEnhance.Sharpness(base).enhance(1.8).convert("RGB")

    gray = ImageOps.grayscale(base)
    gray = ImageOps.autocontrast(gray, cutoff=0.5)
    gray = ImageEnhance.Contrast(gray).enhance(2.2)
    gray = gray.filter(ImageFilter.UnsharpMask(radius=1.6, percent=220, threshold=2))

    threshold = gray.point(lambda p: 255 if p > 150 else 0)
    threshold = threshold.filter(ImageFilter.MedianFilter(size=3))

    return {
        "id_crop_original": original,
        "id_crop_grayscale": gray.convert("RGB"),
        "id_crop_threshold": threshold.convert("RGB"),
    }


def normalized_digits(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = re.sub(r"\D", "", value.translate(ARABIC_TO_LATIN))
    return cleaned or None


def structurally_valid_national_id(value: str | None) -> bool:
    cleaned = normalized_digits(value)
    if not cleaned or len(cleaned) != 14 or cleaned[0] not in {"2", "3"}:
        return False
    try:
        century = 1900 if cleaned[0] == "2" else 2000
        birth = date(century + int(cleaned[1:3]), int(cleaned[3:5]), int(cleaned[5:7]))
        return birth <= date.today()
    except ValueError:
        return False


def vote_national_id(reads: NationalIdReads) -> str | None:
    candidates = [
        normalized_digits(reads.original_read),
        normalized_digits(reads.grayscale_read),
        normalized_digits(reads.threshold_read),
    ]
    valid = [item for item in candidates if structurally_valid_national_id(item)]
    if not valid:
        return None
    counts = Counter(valid)
    value, votes = counts.most_common(1)[0]
    return value if votes >= 2 else None


def read_national_id(key: str, crops: dict[str, Image.Image]) -> NationalIdReads:
    content = [
        {"type": "input_text", "text": "اقرأ كل نسخة مستقلة ثم أرجع القراءات الثلاث."},
        {"type": "input_text", "text": "النسخة الأصلية المكبرة:"},
        {"type": "input_image", "image_url": image_to_data_url(crops["id_crop_original"]), "detail": "high"},
        {"type": "input_text", "text": "النسخة الرمادية عالية التباين:"},
        {"type": "input_image", "image_url": image_to_data_url(crops["id_crop_grayscale"]), "detail": "high"},
        {"type": "input_text", "text": "نسخة الأبيض والأسود:"},
        {"type": "input_image", "image_url": image_to_data_url(crops["id_crop_threshold"]), "detail": "high"},
    ]
    client = OpenAI(api_key=key)
    response = client.responses.parse(
        model=os.getenv("OPENAI_MODEL", "gpt-5.1"),
        instructions=NATIONAL_ID_PROMPT,
        input=[{"role": "user", "content": content}],
        text_format=NationalIdReads,
    )
    if response.output_parsed is None:
        raise RuntimeError("تعذر قراءة الرقم القومي")
    return response.output_parsed


def analyze_documents(key: str, insurance_file, id_file):
    insurance_original = open_uploaded(insurance_file) if insurance_file else None
    id_original = open_uploaded(id_file) if id_file else None
    prepared: dict[str, Image.Image] = {}
    content: list[dict] = [{
        "type": "input_text",
        "text": f"حلل المستندات التالية. تاريخ اليوم {date.today().isoformat()}.",
    }]

    if insurance_original:
        insurance_enhanced = enhance_document(insurance_original)
        prepared["insurance_enhanced"] = insurance_enhanced
        content.extend([
            {"type": "input_text", "text": "كارنيه التأمين الأصلي:"},
            {"type": "input_image", "image_url": image_to_data_url(insurance_original), "detail": "high"},
            {"type": "input_text", "text": "كارنيه التأمين المحسن:"},
            {"type": "input_image", "image_url": image_to_data_url(insurance_enhanced), "detail": "high"},
        ])

    id_reads = None
    verified_national_id = None
    if id_original:
        id_enhanced = enhance_document(id_original)
        crops = national_id_crops(id_original)
        prepared["id_enhanced"] = id_enhanced
        prepared.update(crops)
        content.extend([
            {"type": "input_text", "text": "بطاقة الرقم القومي الأصلية؛ استخرج الاسم فقط ولا تقرأ الرقم القومي هنا:"},
            {"type": "input_image", "image_url": image_to_data_url(id_original), "detail": "high"},
        ])
        id_reads = read_national_id(key, crops)
        verified_national_id = vote_national_id(id_reads)

    client = OpenAI(api_key=key)
    response = client.responses.parse(
        model=os.getenv("OPENAI_MODEL", "gpt-5.1"),
        instructions=PROMPT,
        input=[{"role": "user", "content": content}],
        text_format=ReaderResult,
    )
    if response.output_parsed is None:
        raise RuntimeError("لم يتم استخراج نتيجة منظمة")

    result = response.output_parsed
    result.national_id.value = verified_national_id
    result.national_id.confidence = id_reads.confidence if verified_national_id and id_reads else 0
    if not verified_national_id and id_original:
        result.warnings.append("لم تتفق نسختان من أصل ثلاث على الرقم القومي؛ تُرك فارغًا بدل عرض رقم خاطئ")
    return result, prepared, id_reads


st.set_page_config(page_title="Hekma AI V2.1", page_icon="🧠", layout="wide")
st.markdown(
    "<style>html,body,[class*='css']{direction:rtl;text-align:right}.block-container{max-width:1150px;padding-top:2rem}</style>",
    unsafe_allow_html=True,
)
st.title("🧠 Hekma AI — Vision Pipeline V2.1")
st.caption("ثلاث معالجات مستقلة للرقم القومي، ولا يتم اعتماده إلا بتطابق قراءتين على الأقل.")

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

if st.button("تحليل V2.1", type="primary", use_container_width=True):
    if not api_key():
        st.error("مفتاح OpenAI غير موجود")
    elif not insurance_file and not id_file:
        st.error("ارفع مستندًا واحدًا على الأقل")
    else:
        try:
            with st.spinner("جاري تجهيز ثلاث نسخ للرقم القومي وتحليل المستندات..."):
                result, prepared, id_reads = analyze_documents(api_key(), insurance_file, id_file)
                st.session_state.result_v21 = result
                st.session_state.prepared_v21 = prepared
                st.session_state.id_reads_v21 = id_reads
        except Exception as exc:
            st.error(f"فشل التحليل: {exc}")

result = st.session_state.get("result_v21")
prepared = st.session_state.get("prepared_v21", {})
id_reads = st.session_state.get("id_reads_v21")

if result:
    st.success(result.summary_ar)
    st.subheader({"valid": "🟢 الكارنيه ساري", "expired": "🔴 الكارنيه منتهي", "unknown": "🟡 الصلاحية غير واضحة"}[result.card_status])

    patient_tab, insurance_tab, audit_tab, processing_tab, json_tab = st.tabs([
        "بيانات المريض", "بيانات التأمين", "تدقيق الرقم القومي", "معالجة الصور", "JSON"
    ])

    with patient_tab:
        c1, c2 = st.columns(2)
        c1.text_input("الاسم بالعربي", value=result.patient_name_ar.value or "")
        c2.text_input("الاسم بالإنجليزي", value=result.patient_name_en.value or "")
        st.text_input("الرقم القومي المؤكد", value=result.national_id.value or "")
        if not result.national_id.value:
            st.error("الرقم القومي غير مؤكد؛ لم يتم اعتماد أي تخمين.")

    with insurance_tab:
        c1, c2 = st.columns(2)
        c1.text_input("شركة التأمين", value=result.insurance_company.value or "")
        c2.text_input("جهة العمل", value=result.employer.value or "")
        c1.text_input("Member ID", value=result.member_id.value or "")
        c2.text_input("Card Number", value=result.card_number.value or "")
        c1.text_input("Policy Number", value=result.policy_number.value or "")
        c2.text_input("الفئة / الشبكة", value=result.network_class.value or "")
        st.text_input("تاريخ الانتهاء", value=result.expiry_date.value or "")

    with audit_tab:
        if id_reads:
            st.dataframe({
                "المعالجة": ["الأصلية المكبرة", "رمادية عالية التباين", "أبيض وأسود"],
                "القراءة": [id_reads.original_read, id_reads.grayscale_read, id_reads.threshold_read],
            }, use_container_width=True)
            st.caption(id_reads.note or "")
        else:
            st.info("لم يتم رفع بطاقة رقم قومي")

    with processing_tab:
        if prepared:
            cols = st.columns(min(len(prepared), 3))
            for index, (name, image) in enumerate(prepared.items()):
                with cols[index % len(cols)]:
                    st.image(image, caption=name, use_container_width=True)

    with json_tab:
        st.json({
            "result": result.model_dump(),
            "national_id_reads": id_reads.model_dump() if id_reads else None,
        })

    if result.warnings:
        st.warning("\n".join(f"• {item}" for item in dict.fromkeys(result.warnings)))
