import base64
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
8. أرجع JSON منظم فقط.
"""


def data_url(uploaded_file) -> str:
    encoded = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
    return f"data:{uploaded_file.type};base64,{encoded}"


def analyze(api_key: str, files) -> CardReaderResult:
    client = OpenAI(api_key=api_key)
    content = [{"type": "input_text", "text": f"تاريخ اليوم: {date.today().isoformat()}"}]
    for file in files:
        content.append({"type": "input_image", "image_url": data_url(file), "detail": "high"})

    response = client.responses.parse(
        model="gpt-5.6",
        instructions=PROMPT,
        input=[{"role": "user", "content": content}],
        text_format=CardReaderResult,
    )
    if response.output_parsed is None:
        raise RuntimeError("لم يتم استخراج نتيجة منظمة")
    return response.output_parsed


def field(label: str, item: ExtractedField) -> None:
    st.text_input(label, value=item.value or "غير واضح", help=f"الثقة {item.confidence}% | المصدر: {item.source or 'غير محدد'}")


st.set_page_config(page_title="Hekma Card Reader", page_icon="🪪", layout="wide")
st.markdown("<style>html,body,[class*='css']{direction:rtl;text-align:right}.block-container{max-width:1100px;padding-top:2rem}</style>", unsafe_allow_html=True)

st.title("🪪 Hekma Smart Card Reader")
st.caption("ارفع كارنيه التأمين والبطاقة الشخصية")

with st.sidebar:
    api_key = st.text_input("OpenAI API Key", type="password")

files = st.file_uploader("اختر الصور", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)

if files:
    cols = st.columns(min(len(files), 4))
    for i, file in enumerate(files):
        with cols[i % len(cols)]:
            st.image(file, caption=file.name, use_container_width=True)

if st.button("قراءة البطاقات", type="primary", use_container_width=True):
    if not api_key:
        st.error("اكتب API Key")
    elif not files:
        st.error("ارفع صورة واحدة على الأقل")
    else:
        try:
            with st.spinner("جاري القراءة..."):
                st.session_state.result = analyze(api_key, files)
        except Exception as exc:
            st.error(str(exc))

result = st.session_state.get("result")
if result:
    status = {"valid": "🟢 الكارنيه ساري", "expired": "🔴 الكارنيه منتهي", "unknown": "🟡 الصلاحية غير واضحة"}[result.card_status]
    st.success(result.summary_ar)
    st.subheader(status)

    tab1, tab2, tab3 = st.tabs(["بيانات المريض", "بيانات التأمين", "JSON"])
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            field("الاسم بالعربي", result.patient_name_ar)
            field("الرقم القومي", result.national_id)
            field("النوع", result.gender)
        with c2:
            field("الاسم بالإنجليزي", result.patient_name_en)
            field("تاريخ الميلاد", result.date_of_birth)
            st.text_input("تطابق الأسماء", value=result.names_match)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            field("شركة التأمين", result.insurance_company)
            field("رقم العميل", result.customer_number)
            field("رقم الكارنيه", result.card_number)
            field("جهة العمل", result.employer)
        with c2:
            field("Member ID", result.member_id)
            field("رقم الوثيقة", result.policy_number)
            field("الفئة / الشبكة", result.network_class)
            field("تاريخ البداية", result.valid_from)
            field("تاريخ الانتهاء", result.expiry_date)

    with tab3:
        st.json(result.model_dump())

    if result.warnings:
        st.warning("\n".join(f"• {w}" for w in result.warnings))
