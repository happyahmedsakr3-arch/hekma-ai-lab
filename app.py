import base64
import os
import re
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


class CriticalNumbers(BaseModel):
    national_id: str | None = None
    member_id: str | None = None
    card_number: str | None = None
    national_id_confidence: int = Field(default=0, ge=0, le=100)
    member_id_confidence: int = Field(default=0, ge=0, le=100)
    card_number_confidence: int = Field(default=0, ge=0, le=100)
    notes: list[str] = []


PROMPT = """
أنت قارئ شديد الدقة لكارنيهات التأمين الطبي وبطاقات الرقم القومي المصرية.

القواعد:
1. لا تخمن. أي قيمة غير واضحة = null.
2. فرّق بين رقم العميل وMember ID ورقم الكارنيه ورقم الوثيقة حسب العنوان المطبوع.
3. استخرج الاسم العربي من البطاقة، والاسم الإنجليزي من كارنيه التأمين.
4. الرقم القومي المصري لا يُقبل إلا إذا كان 14 رقمًا كاملًا وواضحًا.
5. عند قراءة الرقم القومي، انسخه بنفس شكل الأرقام العربية المطبوعة على البطاقة دون تحويلها إلى أرقام إنجليزية.
6. حدّد صلاحية الكارنيه بمقارنة تاريخ الانتهاء بتاريخ اليوم.
7. قارن الاسم العربي والإنجليزي عند وجود المستندين.
8. لكل قيمة أرجع المصدر ودرجة ثقة من 0 إلى 100.
9. راجع كل رقم خانة بخانة، ولا تضف أصفارًا أو تكرر رقمًا غير موجود.
10. لا تعتبر كودًا مطبوعًا أو باركودًا رقم كارنيه دون دليل واضح من الموضع أو العنوان.
11. أرجع نتيجة منظمة فقط.
"""

CRITICAL_PROMPT = """
مهمتك OCR حرفي فقط لثلاثة أرقام حساسة:
1) الرقم القومي المصري الموجود في السطر السفلي الثابت ببطاقة الرقم القومي.
2) Member ID المكتوب بجوار ID أو Member ID على كارنيه التأمين.
3) Card Number وهو الرقم الآخر المطبوع على كارنيه التأمين.

تعليمات الرقم القومي:
- انقله كما هو مطبوع بالأرقام العربية: ٠١٢٣٤٥٦٧٨٩.
- لا تحوّله إلى 0123456789.
- اقرأ الخانات الأربع عشرة واحدة واحدة من اليمين إلى اليسار كما تظهر بصريًا، ثم أرجع السلسلة المطبوعة نفسها.
- تجاهل أي كود لاتيني مثل LG أو باركود أو رقم تسلسلي آخر.

تعليمات جميع الأرقام:
- لا تستنتج، لا تصحح، لا تضف ولا تحذف ولا تكرر رقمًا.
- عند الشك أرجع null بدل التخمين.
- أرجع النتيجة المنظمة فقط.
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

ARABIC_TO_LATIN = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def get_api_key() -> str:
    try:
        key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        key = ""
    return key or os.getenv("OPENAI_API_KEY", "")


def data_url(uploaded_file) -> str:
    encoded = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
    return f"data:{uploaded_file.type};base64,{encoded}"


def image_content(files, text: str) -> list[dict]:
    content = [{"type": "input_text", "text": text}]
    for file in files:
        content.append({"type": "input_image", "image_url": data_url(file), "detail": "high"})
    return content


def analyze(api_key: str, files) -> CardReaderResult:
    client = OpenAI(api_key=api_key)
    response = client.responses.parse(
        model=os.getenv("OPENAI_MODEL", "gpt-5.1"),
        instructions=PROMPT,
        input=[{"role": "user", "content": image_content(files, f"حلل المستندات. تاريخ اليوم: {date.today().isoformat()}")}],
        text_format=CardReaderResult,
    )
    if response.output_parsed is None:
        raise RuntimeError("لم يتم استخراج نتيجة منظمة")
    return response.output_parsed


def read_critical_numbers(api_key: str, files, round_number: int) -> CriticalNumbers:
    client = OpenAI(api_key=api_key)
    response = client.responses.parse(
        model=os.getenv("OPENAI_MODEL", "gpt-5.1"),
        instructions=CRITICAL_PROMPT,
        input=[{
            "role": "user",
            "content": image_content(files, f"قراءة حرفية مستقلة رقم {round_number}. انسخ سطر الرقم القومي العربي كما هو، ثم اقرأ رقمي التأمين."),
        }],
        text_format=CriticalNumbers,
    )
    if response.output_parsed is None:
        raise RuntimeError("تعذر التحقق من الأرقام الحساسة")
    return response.output_parsed


def normalized_digits(value: str | None) -> str | None:
    if not value:
        return None
    translated = value.translate(ARABIC_TO_LATIN)
    cleaned = re.sub(r"\D", "", translated)
    return cleaned or None


def exact_display(value: str | None) -> str | None:
    if not value:
        return None
    return re.sub(r"[\s\-–—]", "", value)


def consensus(first: str | None, second: str | None) -> str | None:
    a = normalized_digits(first)
    b = normalized_digits(second)
    if not a or a != b:
        return None
    return exact_display(first)


def verify_national_id(value: str | None) -> dict:
    info = {"valid_structure": False, "birth_date": None, "age": None, "gender": None, "governorate": None, "note": None}
    national_id = normalized_digits(value)
    if not national_id or len(national_id) != 14:
        info["note"] = "الرقم القومي غير مؤكد أو ليس 14 رقمًا"
        return info
    century = {"2": 1900, "3": 2000}.get(national_id[0])
    if century is None:
        info["note"] = "كود القرن غير صحيح"
        return info
    try:
        birth = date(century + int(national_id[1:3]), int(national_id[3:5]), int(national_id[5:7]))
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


def show_field(label: str, item: ExtractedField, key: str) -> str:
    return st.text_input(label, value=item.value or "", help=f"الثقة {item.confidence}% | المصدر: {item.source or 'غير محدد'}", key=key)


st.set_page_config(page_title="Hekma AI", page_icon="🧠", layout="wide")
st.markdown("<style>html,body,[class*='css']{direction:rtl;text-align:right}.block-container{max-width:1100px;padding-top:2rem}</style>", unsafe_allow_html=True)
st.title("🧠 Hekma AI")
st.subheader("قارئ الكارنيه والبطاقة")
st.caption("لأفضل دقة: ارفع صورة قريبة وواضحة للكارنيه وصورة مستقلة للبطاقة الشخصية")

api_key = get_api_key()
with st.sidebar:
    st.success("API متصل") if api_key else st.error("API غير متصل")

files = st.file_uploader("اختر الصور", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
if files:
    columns = st.columns(min(len(files), 4))
    for index, file in enumerate(files):
        with columns[index % len(columns)]:
            st.image(file, caption=file.name, use_container_width=True)

if st.button("تحليل وتدقيق الأرقام", type="primary", use_container_width=True):
    if not api_key:
        st.error("أضف OpenAI API Key أولًا")
    elif not files:
        st.error("ارفع صورة واحدة على الأقل")
    else:
        try:
            with st.spinner("جاري نسخ الرقم القومي العربي حرفيًا وتدقيق أرقام التأمين..."):
                result = analyze(api_key, files)
                pass1 = read_critical_numbers(api_key, files, 1)
                pass2 = read_critical_numbers(api_key, files, 2)
                st.session_state.result = result
                st.session_state.pass1 = pass1
                st.session_state.pass2 = pass2
        except Exception as exc:
            st.error(f"فشل التحليل: {exc}")

result = st.session_state.get("result")
pass1 = st.session_state.get("pass1")
pass2 = st.session_state.get("pass2")

if result and pass1 and pass2:
    verified_national_id = consensus(pass1.national_id, pass2.national_id)
    verified_member_id = consensus(pass1.member_id, pass2.member_id)
    verified_card_number = consensus(pass1.card_number, pass2.card_number)
    verification = verify_national_id(verified_national_id)

    status = {"valid": "🟢 الكارنيه ساري", "expired": "🔴 الكارنيه منتهي", "unknown": "🟡 الصلاحية غير واضحة"}[result.card_status]
    st.success(result.summary_ar)
    st.subheader(status)

    patient_tab, insurance_tab, audit_tab, json_tab = st.tabs(["بيانات المريض", "بيانات التأمين", "تدقيق الأرقام", "JSON"])

    with patient_tab:
        c1, c2 = st.columns(2)
        with c1:
            show_field("الاسم بالعربي", result.patient_name_ar, "name_ar")
            st.text_input("الرقم القومي كما هو مكتوب بالعربي", value=verified_national_id or "", key="verified_nid")
            st.text_input("النوع", value=verification["gender"] or "غير مؤكد")
        with c2:
            show_field("الاسم بالإنجليزي", result.patient_name_en, "name_en")
            st.text_input("تاريخ الميلاد", value=verification["birth_date"] or "غير مؤكد")
            st.text_input("تطابق الأسماء", value=result.names_match)

    with insurance_tab:
        c1, c2 = st.columns(2)
        with c1:
            show_field("شركة التأمين", result.insurance_company, "company")
            st.text_input("رقم الكارنيه المؤكد", value=verified_card_number or "", key="verified_card")
            show_field("جهة العمل", result.employer, "employer")
        with c2:
            st.text_input("Member ID المؤكد", value=verified_member_id or "", key="verified_member")
            show_field("الفئة / الشبكة", result.network_class, "network")
            show_field("تاريخ الانتهاء", result.expiry_date, "expiry")

    with audit_tab:
        st.write("القراءة الحرفية الأولى مقابل الثانية")
        st.dataframe({
            "الحقل": ["الرقم القومي", "Member ID", "رقم الكارنيه"],
            "القراءة الأولى": [pass1.national_id, pass1.member_id, pass1.card_number],
            "القراءة الثانية": [pass2.national_id, pass2.member_id, pass2.card_number],
            "النتيجة المؤكدة": [verified_national_id, verified_member_id, verified_card_number],
        }, use_container_width=True)
        if not verified_national_id:
            st.error("الرقم القومي العربي لم يتطابق حرفيًا في القراءتين؛ ارفع لقطة أقرب للسطر السفلي فقط")
        if not verified_card_number:
            st.error("رقم الكارنيه لم يتطابق في القراءتين؛ ارفع لقطة أقرب لرقم الكارنيه فقط")
        if verification["valid_structure"]:
            st.success("الرقم القومي المؤكد صالح من حيث البنية والتاريخ")
        else:
            st.warning(verification["note"])

    with json_tab:
        st.json({
            "general": result.model_dump(),
            "critical_pass_1": pass1.model_dump(),
            "critical_pass_2": pass2.model_dump(),
            "verified": {
                "national_id_arabic": verified_national_id,
                "national_id_normalized": normalized_digits(verified_national_id),
                "member_id": verified_member_id,
                "card_number": verified_card_number,
            },
            "national_id_verification": verification,
        })