from pydantic import BaseModel
from typing import Optional


class ConfidenceField(BaseModel):
    value: Optional[str] = None
    confidence: float = 0.0


class CardData(BaseModel):
    company: ConfidenceField
    patient_name_ar: ConfidenceField
    patient_name_en: ConfidenceField
    national_id: ConfidenceField
    member_id: ConfidenceField
    card_number: ConfidenceField
    employer: ConfidenceField
    category: ConfidenceField
    expiry_date: ConfidenceField

    card_valid: bool = False


class DoctorRequest(BaseModel):
    doctor_name: ConfidenceField
    diagnosis: ConfidenceField
    requested_service: ConfidenceField
    handwritten_text: ConfidenceField


class CaseAnalysis(BaseModel):
    card: CardData
    request: Optional[DoctorRequest] = None

    missing_documents: list[str] = []

    warnings: list[str] = []

    overall_confidence: float = 0.0
