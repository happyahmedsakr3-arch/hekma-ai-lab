import os

import streamlit as st

from src.insurance_service import analyze_insurance_card


def api_key() -> str:
    try:
        return st.secrets.get("OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
    except Exception:
        return os.getenv("OPENAI_API_KEY", "")


st.set_page_config(page_title="Hekma AI - Insurance Card Reader", page_icon="🧠", layout="wide")
st.markdown(
    """
    <style>
    html,body,[class*='css']{direction:rtl;text-align:right}
    .block-container{max-width:1100px;padding-top:2rem}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧠 Hekma AI — قارئ كارنيه التأمين")
st.caption("ارفع صورة كارنيه التأمين فقط، وسيتم استخراج بياناته ومراجعتها.")

with st.sidebar:
    st.info("الوضع الحالي: قارئ كارنيه التأمين فقط")
    st.success("GPT متصل") if api_key() else st.warning("GPT غير متصل")

insurance_file = st.file_uploader(
    "ارفع صورة كارنيه التأمين",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=False,
    key="insurance_card_only",
)

if insurance_file:
    st.image(insurance_file, caption="كارنيه التأمين", use_container_width=True)

if st.button("تحليل الكارنيه", type="primary", use_container_width=True):
    if not insurance_file:
        st.error("ارفع صورة كارنيه التأمين أولاً")
    else:
        try:
            with st.spinner("جاري قراءة الكارنيه وتدقيق البيانات..."):
                bundle = analyze_insurance_card(
                    insurance_file.getvalue(),
                    api_key=api_key(),
                    model=os.getenv("OPENAI_MODEL", "gpt-5.1"),
                )
                st.session_state.card_ocr = bundle.get("ocr")
                st.session_state.card_hints = bundle.get("hints", {})
                st.session_state.card_result = bundle.get("result")
        except Exception as exc:
            st.error(f"فشل تحليل الكارنيه: {exc}")

ocr = st.session_state.get("card_ocr")
hints = st.session_state.get("card_hints", {})
result = st.session_state.get("card_result")

if ocr:
    tab1, tab2, tab3 = st.tabs(["النتيجة", "OCR الخام", "التدقيق"])

    with tab1:
        st.subheader("بيانات التأمين")

        if result:
            st.success(result.get("summary_ar", "تم تحليل الكارنيه"))
            a, b = st.columns(2)

            def fv(name):
                field = result.get(name) or {}
                return field.get("value") if isinstance(field, dict) else ""

            a.text_input("الاسم", value=fv("patient_name_en") or fv("patient_name_ar") or hints.get("name") or "")
            b.text_input("شركة التأمين", value=fv("insurance_company") or hints.get("company") or "")
            a.text_input("Card Number", value=fv("card_number") or hints.get("card_number") or "")
            b.text_input("ID No", value=fv("id_number") or hints.get("id_number") or "")
            a.text_input("Member ID", value=fv("member_id") or hints.get("member_id") or "")
            b.text_input("Policy Number", value=fv("policy_number") or hints.get("policy_number") or "")
            a.text_input("جهة العمل / Policy Holder", value=fv("employer") or hints.get("employer") or "")
            b.text_input("الفئة / الشبكة", value=fv("network_class") or hints.get("network_class") or "")
            a.text_input("بداية الصلاحية", value=fv("valid_from") or hints.get("valid_from") or "")
            b.text_input("تاريخ الانتهاء", value=fv("expiry_date") or hints.get("expiry_date") or "")

            status = {
                "valid": "🟢 الكارنيه ساري",
                "expired": "🔴 الكارنيه منتهي",
                "unknown": "🟡 حالة الكارنيه غير مؤكدة",
            }.get(result.get("card_status", "unknown"), "🟡 حالة الكارنيه غير مؤكدة")
            st.subheader(status)
        else:
            st.info("لم يتم تنظيم نتيجة الكارنيه.")
            if hints:
                st.json(hints)

    with tab2:
        st.text_area("نص OCR", value=ocr.get("text", ""), height=320)

    with tab3:
        if hints:
            st.write("الحقول التي استخرجتها القواعد قبل GPT:")
            st.json(hints)
        warnings = (result or {}).get("warnings", []) if isinstance(result, dict) else []
        if warnings:
            st.warning("\n".join("• " + x for x in warnings))
