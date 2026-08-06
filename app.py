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


class CardReaderResult(BaseModel):
    patient_name_ar: ExtractedField
    patient_name_en: ExtractedField
    national_id: ExtractedField
    date_of_birth: ExtractedField
    gender: ExtractedField
    insurance_company: ExtractedField
    customer_number: ExtractedField
    member_id: ExtractedField
    card_number: ExtractedField
    policy_number: ExtractedField
    employer: ExtractedField
    network_class: ExtractedField
    valid_from: ExtractedField
    expiry_date: ExtractedField
    card_status: Literal["valid", "expired", "unknown"]
    names_match: Literal["match", "possible_match", "mismatch", "unknown"]
    warnings: list[str]
    summary_ar: str


PROMPT = """
أنت قارئ شديد الدقة لكارنيهات التأمين الطبي وبطاقات الرقم القومي المصرية.

القواعد:
1. لا تخمن. أي قيمة غير واضحة = null.
2. فرّق بين رقم العميل وMember ID ورقم الكارنيه ورقم الوثيقة حسب العنوان المطبوع.
3. استخرج الاسم العربي من البطاقة، وإن كان الاسم على الكارنيه بالإنجليزية فقط فاكتبه بالعربية كتابة صوتية محافظة.
4. استخرج الرقم القومي فقط إذا كان كاملًا وواضحًا.
5. حدّد صلاحية الكارنيه بمقارنة تاريخ الانتهاء بتاريخ اليوم.
6. قارن اسم البطاقة باسم الكارنيه.
7. لكل قيمة أرجع المصدر ودرجة ثقة من 0 إلى 100.
8. أرجع نتيجة منظمة فقط.
"""


def data_url(uploaded_file) -> str:
    encoded = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
    return f"data:{uploaded_file.type};base64,{encoded}"


def analyze(api_key: str, files) -> CardReaderResult:
    client = OpenAI(api_key=api_key)
    content = [
        {
            "type": "input_text",
            "text": f"حلل المستندات المرفقة. تاريخ اليوم: {date.today().isoformat()}",
        }
    ]

    for file in files:
        content.append(
            {
                "type": "input_image",
                "image_url": data_url(file),
                "detail": "high",
            }
        )

    response = client.responses.parse(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        instructions=PROMPT,
        input=[{"role": "user", "content": content}],
        text_format=CardReaderResult,
    )

    if response.output_parsed is None:
        raise RuntimeError("لم يتم استخراج نتيجة منظمة")

    return response.output_parsed


def show_field(label: str, item: ExtractedField) -> None:
    st.text_input(
        label,
        value=item.value or "غير واضح",
        help=f"الثقة {item.confidence}% | المصدر: {item.source or 'غير محدد'}",
    )


st.set_page_config(page_title="Hekma AI", page_icon="🧠", layout="wide")
st.markdown(
    """
    <style>
    html, body, [class*='css'] { direction: rtl; text-align: right; }
    .block-container { max-width: 1100px; padding-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧠 Hekma AI")
st.subheader("قارئ الكارنيه والبطاقة")
st.caption("ارفع صورة الكارنيه وصورة البطاقة ثم اضغط تحليل")

with st.sidebar:
    api_key = st.text_input("OpenAI API Key", type="password")
    st.caption("المفتاح لا يُحفظ داخل GitHub")

files = st.file_uploader(
    "اختر الصور",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True,
)

if files:
    columns = st.columns(min(len(files), 4))
    for index, file in enumerate(files):
        with columns[index % len(columns)]:
            st.image(file, caption=file.name, use_container_width=True)

if st.button("تحليل البطاقات", type="primary", use_container_width=True):
    if not api_key:
        st.error("اكتب API Key")
    elif not files:
        st.error("ارفع صورة واحدة على الأقل")
    else:
        try:
            with st.spinner("جاري قراءة البيانات..."):
                st.session_state.result = analyze(api_key, files)
        except Exception as exc:
            st.error(f"فشل التحليل: {exc}")

result = st.session_state.get("result")
if result:
    status = {
        "valid": "🟢 الكارنيه ساري",
        "expired": "🔴 الكارنيه منتهي",
        "unknown": "🟡 الصلاحية غير واضحة",
    }[result.card_status]

    st.success(result.summary_ar)
    st.subheader(status)

    patient_tab, insurance_tab, json_tab = st.tabs(
        ["بيانات المريض", "بيانات التأمين", "JSON"]
    )

    with patient_tab:
        col1, col2 = st.columns(2)
        with col1:
            show_field("الاسم بالعربي", result.patient_name_ar)
            show_field("الرقم القومي", result.national_id)
            show_field("النوع", result.gender)
        with col2:
            show_field("الاسم بالإنجليزي", result.patient_name_en)
            show_field("تاريخ الميلاد", result.date_of_birth)
            st.text_input("تطابق الأسماء", value=result.names_match)

    with insurance_tab:
        col1, col2 = st.columns(2)
        with col1:
            show_field("شركة التأمين", result.insurance_company)
            show_field("رقم العميل", result.customer_number)
            show_field("رقم الكارنيه", result.card_number)
            show_field("جهة العمل", result.employer)
        with col2:
            show_field("Member ID", result.member_id)
            show_field("رقم الوثيقة", result.policy_number)
            show_field("الفئة / الشبكة", result.network_class)
            show_field("تاريخ البداية", result.valid_from)
            show_field("تاريخ الانتهاء", result.expiry_date)

    with json_tab:
        st.json(result.model_dump())

    if result.warnings:
        st.warning("\n".join(f"• {warning}" for warning in result.warnings))
