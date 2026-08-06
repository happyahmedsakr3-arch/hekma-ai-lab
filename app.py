import base64
import os
from datetime import date
from typing import Literal

import streamlit as st
from openai import OpenAI
from pydantic import BaseModel, Field


class ExtractedField(BaseModel):
    value: str | None = None
    confidence: int = Field(default=0, ge=0, le=100)
    source: str | None = None


class PatientData(BaseModel):
    arabic_name: ExtractedField
    english_name: ExtractedField
    national_id: ExtractedField
    date_of_birth: ExtractedField
    gender: ExtractedField


class InsuranceData(BaseModel):
    company: ExtractedField
    customer_number: ExtractedField
    member_id: ExtractedField
    card_number: ExtractedField
    policy_number: ExtractedField
    network_class: ExtractedField
    expiry_date: ExtractedField
    card_status: Literal["valid", "expired", "unknown"]


class DoctorRequest(BaseModel):
    doctor_name: ExtractedField
    specialty: ExtractedField
    diagnosis: ExtractedField
    requested_service: ExtractedField
    handwritten_text: ExtractedField
    unclear_words: list[str]


class CaseResult(BaseModel):
    patient: PatientData
    insurance: InsuranceData
    doctor_request: DoctorRequest
    warnings: list[str]
    summary_ar: str


SYSTEM_PROMPT = """
أنت Hekma AI، مساعد متخصص في قراءة مستندات الموافقات الطبية المصرية.
حلّل الصور المرفوعة باعتبارها قد تشمل كارنيه تأمين، بطاقة رقم قومي، وطلب طبيب بخط اليد.

القواعد:
1) لا تخمّن. أي معلومة غير واضحة اجعلها null.
2) اكتب الاسم العربي بالعربية إذا كان ممكنًا استخلاصه بأمان من البطاقة أو الاسم الإنجليزي؛ إن لم تكن متأكدًا اجعله null.
3) فرّق بين رقم العميل، Member ID، رقم الكارنيه، ورقم الوثيقة حسب العناوين المطبوعة على المستند.
4) استخرج الرقم القومي المصري من البطاقة فقط عندما يكون واضحًا بالكامل.
5) حدّد صلاحية الكارنيه بمقارنة تاريخ الانتهاء بتاريخ اليوم المرسل لك.
6) اقرأ خط الطبيب قدر الإمكان، وحدد التشخيص والخدمة المطلوبة والكلمات غير الواضحة.
7) لكل قيمة أرجع المصدر ودرجة ثقة من 0 إلى 100.
8) لا تضف أعراضًا أو تشخيصات غير موجودة.
9) اكتب ملخصًا عربيًا قصيرًا وواضحًا.
"""


def file_to_data_url(uploaded_file) -> str:
    encoded = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
    return f"data:{uploaded_file.type};base64,{encoded}"


def analyze_case(api_key: str, uploaded_files) -> CaseResult:
    client = OpenAI(api_key=api_key)
    content = [
        {
            "type": "input_text",
            "text": f"حلّل هذه المستندات. تاريخ اليوم: {date.today().isoformat()}",
        }
    ]

    for uploaded_file in uploaded_files:
        content.append(
            {
                "type": "input_image",
                "image_url": file_to_data_url(uploaded_file),
                "detail": "high",
            }
        )

    response = client.responses.parse(
        model=os.getenv("OPENAI_MODEL", "gpt-5.6"),
        instructions=SYSTEM_PROMPT,
        input=[{"role": "user", "content": content}],
        text_format=CaseResult,
    )

    if response.output_parsed is None:
        raise RuntimeError("لم يتم إرجاع نتيجة منظمة من النموذج.")

    return response.output_parsed


def show_field(label: str, field: ExtractedField) -> None:
    value = field.value or "غير واضح"
    st.text_input(
        label,
        value=value,
        help=f"الثقة: {field.confidence}% | المصدر: {field.source or 'غير محدد'}",
    )


st.set_page_config(page_title="Hekma AI Lab", page_icon="🧠", layout="wide")
st.markdown(
    """
    <style>
    html, body, [class*="css"] { direction: rtl; text-align: right; }
    .block-container { max-width: 1200px; padding-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧠 Hekma AI Lab")
st.caption("قارئ الكارنيه والبطاقة وطلب الطبيب — نسخة التجارب الأولى")

with st.sidebar:
    st.header("الإعدادات")
    api_key = st.text_input("OpenAI API Key", type="password")
    st.caption("المفتاح يُستخدم داخل الجلسة فقط ولا يُحفظ في GitHub.")

uploaded_files = st.file_uploader(
    "ارفع صور الكارنيه والبطاقة وطلب الطبيب",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True,
)

if uploaded_files:
    preview_columns = st.columns(min(len(uploaded_files), 4))
    for index, uploaded_file in enumerate(uploaded_files):
        with preview_columns[index % len(preview_columns)]:
            st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)

analyze_clicked = st.button("تحليل الحالة", type="primary", use_container_width=True)

if analyze_clicked:
    if not api_key:
        st.error("اكتب API Key أولًا.")
    elif not uploaded_files:
        st.error("ارفع صورة واحدة على الأقل.")
    else:
        try:
            with st.spinner("جاري قراءة المستندات وتحليلها..."):
                result = analyze_case(api_key, uploaded_files)
            st.session_state["result"] = result
        except Exception as exc:
            st.error(f"فشل التحليل: {exc}")

result: CaseResult | None = st.session_state.get("result")
if result:
    st.success(result.summary_ar)

    patient_tab, insurance_tab, doctor_tab, raw_tab = st.tabs(
        ["بيانات المريض", "بيانات التأمين", "طلب الطبيب", "النتيجة الكاملة"]
    )

    with patient_tab:
        col1, col2 = st.columns(2)
        with col1:
            show_field("الاسم بالعربي", result.patient.arabic_name)
            show_field("الرقم القومي", result.patient.national_id)
            show_field("النوع", result.patient.gender)
        with col2:
            show_field("الاسم بالإنجليزي", result.patient.english_name)
            show_field("تاريخ الميلاد", result.patient.date_of_birth)

    with insurance_tab:
        status_text = {
            "valid": "🟢 الكارنيه ساري",
            "expired": "🔴 الكارنيه منتهي",
            "unknown": "🟡 صلاحية الكارنيه غير واضحة",
        }[result.insurance.card_status]
        st.subheader(status_text)
        col1, col2 = st.columns(2)
        with col1:
            show_field("شركة التأمين", result.insurance.company)
            show_field("رقم العميل", result.insurance.customer_number)
            show_field("رقم الكارنيه", result.insurance.card_number)
            show_field("الفئة / الشبكة", result.insurance.network_class)
        with col2:
            show_field("Member ID", result.insurance.member_id)
            show_field("رقم الوثيقة", result.insurance.policy_number)
            show_field("تاريخ الانتهاء", result.insurance.expiry_date)

    with doctor_tab:
        show_field("اسم الطبيب", result.doctor_request.doctor_name)
        show_field("التخصص", result.doctor_request.specialty)
        show_field("التشخيص", result.doctor_request.diagnosis)
        show_field("الخدمة المطلوبة", result.doctor_request.requested_service)
        show_field("النص المكتوب بخط اليد", result.doctor_request.handwritten_text)
        if result.doctor_request.unclear_words:
            st.warning("كلمات غير واضحة: " + " — ".join(result.doctor_request.unclear_words))

    with raw_tab:
        st.json(result.model_dump())

    if result.warnings:
        st.warning("\n".join(f"• {warning}" for warning in result.warnings))
