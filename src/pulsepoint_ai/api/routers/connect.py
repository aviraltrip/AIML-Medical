from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List
from pulsepoint_ai.engines.connect import medreach, translation
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
