from pydantic import BaseModel

class CardData(BaseModel):
    company: str | None = None
    patient_name_ar: str | None = None
    patient_name_en: str | None = None
    national_id: str | None = None
    member_id: str | None = None
    customer_number: str | None = None
    card_number: str | None = None
    policy_number: str | None = None
    expiry_date: str | None = None
    card_valid: bool | None = None

class DoctorRequest(BaseModel):
    doctor_name: str | None = None
    diagnosis: str | None = None
    requested_service: str | None = None
    handwritten_text: str | None = None

class CaseAnalysis(BaseModel):
    card: CardData
    request: DoctorRequest
    missing_documents: list[str] = []
    warnings: list[str] = []
