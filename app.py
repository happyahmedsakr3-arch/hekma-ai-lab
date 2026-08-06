import base64
import os
from datetime import date, datetime
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
3. استخرج الاسم العربي من البطاقة. إذا كان الاسم بالإنجليزية فقط، اكتبه بالعربية كتابة صوتية محافظة مع درجة ثقة مناسبة.
4. استخرج الرقم القومي المصري فقط إذا كان كاملًا وواضحًا من 14 رقمًا.
5. حدّد صلاحية الكارنيه بمقارنة تاريخ الانتهاء بتاريخ اليوم.
6. قارن اسم البطاقة باسم الكارنيه عند وجود الصورتين.
7. لكل قيمة أرجع المصدر ودرجة ثقة من 0 إلى 100.
8. لا تعتبر الباركود أو الأرقام الصغيرة رقم كارنيه إلا إذا دلّ موضعها أو عنوانها على ذلك.
9. راجع الأرقام الحساسة بصريًا مرتين قبل الإرجاع، خصوصًا رقم الكارنيه والرقم القومي.
10. أرجع نتيجة منظمة فقط.
"""


GOVERNORATES = {
    "01": "القاهرة", "02": "الإسكندرية", "03": "بورسعيد", "04": "السويس",
    "11": "دمياط", "12": "الدقهلية", "13": "الشرقية", "14": "القليوبية",
    "15": "كفر الشيخ", "16": "الغربية", "17": "المنوفية", "18": "البحيرة",
    "19": "الإسماعيلية", "21": "الجيزة", "22": "بني سويف", "23": "الفيوم",
    "24": "المنيا", "25": "أسيوط", "26": "سوهاج", "27": "قنا",
    "28": "أسوان", "29": "الأقصر", "31": "البحر الأحمر", "32": "الوادي الجديد",
    "33": "مطروح", "34": "شمال سيناء", "35": "جنوب سيناء", "88": "خارج الجمهورية",
}


def get_api_key() -> str:
    try:
        secret_key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        secret_key = ""
    return secret_key or os.getenv("OPENAI_API_KEY", "")


def data_url(uploaded_file) -> str:
    encoded = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
    return f"data:{uploaded_file.type};base64,{encoded}"


def analyze(api_key: str, files) -> CardReaderResult:
    client = OpenAI(api_key=api_key)
    content = [{"type": "input_text", "text": f"حلل المستندات المرفقة. تاريخ اليوم: {date.today().isoformat()}"}]

    for file in files:
        content.append({"type": "input_image", "image_url": data_url(file), "detail": "high"})

    response = client.responses.parse(
        model=os.getenv("OPENAI_MODEL", "gpt-5.1"),
        instructions=PROMPT,
        input=[{"role": "user", "content": content}],
        text_format=CardReaderResult,
    )

    if response.output_parsed is None:
        raise RuntimeError("لم يتم استخراج نتيجة منظمة")
    return response.output_parsed


def verify_national_id(value: str | None) -> dict:
    info = {"valid_structure": False, "birth_date": None, "age": None, "gender": None, "governorate": None, "note": None}
    if not value:
        info["note"] = "الرقم القومي غير متاح"
        return info

    national_id = "".join(ch for ch in value if ch.isdigit())
    if len(national_id) != 14:
        info["note"] = "الرقم القومي يجب أن يكون 14 رقمًا"
        return info

    century = {"2": 1900, "3": 2000}.get(national_id[0])
    if century is None:
        info["note"] = "كود القرن غير صحيح"
        return info

    try:
        birth = date(
            century + int(national_id[1:3]),
            int(national_id[3:5]),
            int(national_id[5:7]),
        )
    except ValueError:
        info["note"] = "تاريخ الميلاد داخل الرقم القومي غير صحيح"
        return info

    governorate = GOVERNORATES.get(national_id[7:9])
    if governorate is None:
        info["note"] = "كود المحافظة غير معروف"
        return info

    today = date.today()
    age = today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
    gender = "ذكر" if int(national_id[12]) % 2 else "أنثى"

    info.update({
        "valid_structure": True,
        "birth_date": birth.strftime("%d/%m/%Y"),
        "age": age,
        "gender": gender,
        "governorate": governorate,
        "note": "تم التحقق من البنية والتاريخ وكود المحافظة",
    })
    return info


def overall_confidence(result: CardReaderResult) -> int:
    fields = [
        result.patient_name_ar, result.patient_name_en, result.national_id,
        result.insurance_company, result.member_id, result.card_number,
        result.employer, result.network_class, result.expiry_date,
    ]
    scores = [field.confidence for field in fields if field.value]
    return round(sum(scores) / len(scores)) if scores else 0


def show_field(label: str, item: ExtractedField) -> None:
    st.text_input(label, value=item.value or "غير واضح", help=f"الثقة {item.confidence}% | المصدر: {item.source or 'غير محدد'}")


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

saved_api_key = get_api_key()
with st.sidebar:
    if saved_api_key:
        st.success("API متصل")
        api_key = saved_api_key
    else:
        api_key = st.text_input("OpenAI API Key", type="password")
        st.caption("المفتاح يُستخدم داخل الجلسة فقط")

files = st.file_uploader("اختر الصور", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)

if files:
    columns = st.columns(min(len(files), 4))
    for index, file in enumerate(files):
        with columns[index % len(columns)]:
            st.image(file, caption=file.name, use_container_width=True)

if st.button("تحليل البطاقات", type="primary", use_container_width=True):
    if not api_key:
        st.error("أضف OpenAI API Key أولًا")
    elif not files:
        st.error("ارفع صورة واحدة على الأقل")
    else:
        try:
            with st.spinner("جاري قراءة البيانات والتحقق منها..."):
                st.session_state.result = analyze(api_key, files)
        except Exception as exc:
            st.error(f"فشل التحليل: {exc}")

result = st.session_state.get("result")
if result:
    verification = verify_national_id(result.national_id.value)
    confidence = overall_confidence(result)
    status = {"valid": "🟢 الكارنيه ساري", "expired": "🔴 الكارنيه منتهي", "unknown": "🟡 الصلاحية غير واضحة"}[result.card_status]

    st.success(result.summary_ar)
    col_status, col_conf = st.columns(2)
    with col_status:
        st.subheader(status)
    with col_conf:
        st.metric("نسبة الثقة العامة", f"{confidence}%")
        st.progress(confidence / 100)

    patient_tab, insurance_tab, verification_tab, json_tab = st.tabs(
        ["بيانات المريض", "بيانات التأمين", "التحقق الذكي", "JSON"]
    )

    with patient_tab:
        col1, col2 = st.columns(2)
        with col1:
            show_field("الاسم بالعربي", result.patient_name_ar)
            show_field("الرقم القومي", result.national_id)
            st.text_input("النوع", value=verification["gender"] or result.gender.value or "غير واضح")
        with col2:
            show_field("الاسم بالإنجليزي", result.patient_name_en)
            st.text_input("تاريخ الميلاد", value=verification["birth_date"] or result.date_of_birth.value or "غير واضح")
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

    with verification_tab:
        if verification["valid_structure"]:
            st.success("الرقم القومي سليم من حيث البنية والتاريخ وكود المحافظة")
        else:
            st.warning(verification["note"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("تاريخ الميلاد", verification["birth_date"] or "—")
        c2.metric("العمر", str(verification["age"]) if verification["age"] is not None else "—")
        c3.metric("النوع", verification["gender"] or "—")
        c4.metric("المحافظة", verification["governorate"] or "—")

        ready = (
            result.card_status == "valid"
            and result.names_match in {"match", "possible_match"}
            and verification["valid_structure"]
            and confidence >= 75
        )
        if ready:
            st.success("قرار Hekma AI: البيانات صالحة مبدئيًا لإنشاء طلب موافقة")
        else:
            st.warning("قرار Hekma AI: راجع التحذيرات أو البيانات منخفضة الثقة قبل إنشاء طلب الموافقة")

    with json_tab:
        st.json({"extracted": result.model_dump(), "verification": verification, "overall_confidence": confidence})

    warnings = list(result.warnings)
    if result.card_number.confidence < 80:
        warnings.append("رقم الكارنيه منخفض الثقة؛ يُفضّل مراجعته يدويًا")
    if not verification["valid_structure"] and result.national_id.value:
        warnings.append(verification["note"] or "تعذر التحقق من الرقم القومي")
    if warnings:
        st.warning("\n".join(f"• {warning}" for warning in dict.fromkeys(warnings)))
