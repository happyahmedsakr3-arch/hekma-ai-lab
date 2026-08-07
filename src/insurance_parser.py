import re
from datetime import datetime


def _clean(value: str | None) -> str | None:
    if not value:
        return None
    value = re.sub(r"\s+", " ", value).strip(" :|-\n\t")
    return value or None


def detect_company(text: str) -> str | None:
    upper = text.upper()
    if "AXA" in upper:
        return "AXA"
    if "NEXTCARE" in upper or "NEXT CARE" in upper:
        return "NextCare"
    if "GLOBEMED" in upper or "GLOBE MED" in upper:
        return "GlobeMed Egypt"
    if "MEDRIGHT" in upper or "MED RIGHT" in upper:
        return "MedRight"
    if "METLIFE" in upper:
        return "MetLife"
    if "BUPA" in upper:
        return "BUPA"
    return None


def _first(text: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.I | re.M)
        if m:
            return _clean(m.group(1))
    return None


def parse_labelled_fields(text: str) -> dict:
    fields = {
        "company": detect_company(text),
        "name": _first(text, [
            r"(?:^|\n)\s*Name\s*[:\-]\s*([^\n]+)",
            r"(?:^|\n)\s*Member\s*Name\s*[:\-]\s*([^\n]+)",
        ]),
        "card_number": _first(text, [
            r"Card\s*(?:Number|No\.?|#)\s*[:\-]\s*([A-Z0-9\-]+)",
            r"Card\s*No\.?\s*[:\-]\s*([A-Z0-9\-]+)",
        ]),
        "id_number": _first(text, [
            r"(?:^|\n)\s*ID\s*(?:No\.?|Number)?\s*[:\-]\s*([0-9٠-٩۰-۹]{10,20})",
            r"National\s*ID\s*[:\-]\s*([0-9٠-٩۰-۹]{10,20})",
        ]),
        "member_id": _first(text, [
            r"Member\s*ID\s*[:\-]\s*([A-Z0-9\-]+)",
            r"Membership\s*ID\s*[:\-]\s*([A-Z0-9\-]+)",
        ]),
        "policy_number": _first(text, [
            r"Policy\s*(?:No\.?|Number)\s*[:\-]\s*([A-Z0-9\-]+)",
        ]),
        "employer": _first(text, [
            r"Policy\s*Holder\s*[:\-]\s*([^\n]+)",
            r"Company\s*[:\-]\s*([^\n]+)",
            r"Employer\s*[:\-]\s*([^\n]+)",
        ]),
        "network_class": _first(text, [
            r"(?:Network|Category|Class)\s*[:\-]\s*([^\n]+)",
        ]),
        "valid_from": _first(text, [
            r"Valid\s*From\s*[:\-]\s*([0-9A-Za-z\-/]+)",
            r"Effective\s*[:\-]\s*([0-9A-Za-z\-/]+)",
        ]),
        "expiry_date": _first(text, [
            r"Valid\s*To\s*[:\-]\s*([0-9A-Za-z\-/]+)",
            r"Expiry\s*[:\-]\s*([0-9A-Za-z\-/]+)",
            r"Valid\s*Until\s*[:\-]\s*([0-9A-Za-z\-/]+)",
        ]),
    }
    return fields


def normalise_date(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip()
    for fmt in ("%d-%b-%y", "%d-%b-%Y", "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            pass
    return value


def build_hints(text: str) -> dict:
    result = parse_labelled_fields(text)
    result["valid_from"] = normalise_date(result.get("valid_from"))
    result["expiry_date"] = normalise_date(result.get("expiry_date"))
    return result
