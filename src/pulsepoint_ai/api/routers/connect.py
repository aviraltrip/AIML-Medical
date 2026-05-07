from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List
from pulsepoint_ai.engines.connect import care_locator, medreach, translation
from pulsepoint_ai.core.schemas.care import CareLocatorRequest, CareLocatorResponse
from pulsepoint_ai.llm.client import LLMClient
import uuid

router = APIRouter()
llm_client = LLMClient()

class MedReachRequest(BaseModel):
    patient_data: Dict[str, Any]
    triage_history: List[Dict[str, Any]]

class TranslationRequest(BaseModel):
    text: str
    target_language: str

@router.post("/summarize")
async def summarize_case(request: MedReachRequest):
    """
    Generate a clinical summary for a doctor handover based on patient data and triage history.
    """
    try:
        summary = await medreach.summarize_for_doctor(
            request.patient_data, 
            request.triage_history, 
            llm=llm_client
        )
        return {
            "request_id": str(uuid.uuid4()),
            "summary": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/translate")
async def translate_text(request: TranslationRequest):
    """
    Translate medical text into a target language (e.g., 'hi' for Hindi, 'ta' for Tamil).
    """
    try:
        translated = await translation.translate_medical_text(
            request.text,
            request.target_language,
            llm=llm_client
        )
        return {
            "request_id": str(uuid.uuid4()),
            "translated_text": translated,
            "target_language": request.target_language
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/care-locator", response_model=CareLocatorResponse)
async def find_nearby_care(request: CareLocatorRequest):
    """
    HyperLocal Care Matching.

    Takes ICD-10 codes from /api/v1/predict/disease and the severity tier
    from /api/v1/triage/assess plus the patient's coordinates, and returns
    a ranked list of nearby doctors.

    Pipeline:
      1. Multi-label ICD-10 -> specialty mapper (config-driven, longest-prefix).
      2. Haversine geo filter using patient_lat/patient_lon and radius_km.
      3. Weighted relevance scorer (specialty / proximity / availability / rating).
      4. Urgency-conditioned sort (EMERGENCY/URGENT -> distance + same-day only).
    """
    try:
        return care_locator.find_care(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
