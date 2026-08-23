import uuid

from fastapi import APIRouter, HTTPException

from pulsepoint_ai.core.schemas.lab import LabAnalyzeRequest, LabAnalyzeResponse
from pulsepoint_ai.core.schemas.predict import (
    ConditionCardRequest,
    ConditionCardResponse,
    DiseasePredictRequest,
    DiseasePredictResponse,
    SymptomExtractionRequest,
    SymptomExtractionResponse,
)
from pulsepoint_ai.engines.predict import (
    condition_cards,
    lab_detector,
    lab_explainer,
    symptom_extractor,
)
from pulsepoint_ai.engines.predict.disease_classifier import infer as disease_infer
from pulsepoint_ai.llm.client import LLMClient

router = APIRouter()
llm_client = LLMClient()

@router.post("/disease", response_model=DiseasePredictResponse)
async def predict_disease(request: DiseasePredictRequest):
    """
    Predict top likely diseases (ICD-10) based on symptoms, age, and gender.
    """
    try:
        results = disease_infer.predict(
            request.symptoms,
            age=request.age,
            gender=request.gender,
            top_k=request.top_k
        )
        return DiseasePredictResponse(
            request_id=str(uuid.uuid4()),
            predictions=results["predictions"],
            model_version=results["model_version"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@router.post("/labs", response_model=LabAnalyzeResponse)
async def analyze_labs(request: LabAnalyzeRequest):
    """
    Detect lab results from OCR text and provide AI-powered explanations for abnormal values.
    """
    try:

        flags, unflagged_count = lab_detector.detect_labs(
            request.ocr_text,
            age=request.age,
            gender=request.gender
        )


        annotated_flags, blocked_ids = await lab_explainer.explain_flags(
            flags,
            llm=llm_client
        )

        return LabAnalyzeResponse(
            request_id=str(uuid.uuid4()),
            flags=annotated_flags,
            unflagged_count=unflagged_count,
            hallucination_check={
                "blocked": blocked_ids,
                "passed": len(blocked_ids) == 0
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@router.post("/condition-card", response_model=ConditionCardResponse)
async def get_condition_card(request: ConditionCardRequest):
    """
    Generate a plain-English 'Condition Card' with summaries and action steps for an ICD-10 code.
    """
    try:
        response = await condition_cards.get_card(request, llm=llm_client)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/symptoms-from-text", response_model=SymptomExtractionResponse)
async def extract_symptoms_from_text(request: SymptomExtractionRequest):
    """
    Extract canonical symptom names from PDF/OCR/clinical-note text.

    Negation-aware: phrases inside negation scopes ("no history of X",
    "denies Y", "absent Z", etc.) are reported separately under "negated"
    and are NOT included in the positive "symptoms" list. This prevents
    naive substring matchers from emitting "blood cough" as a symptom when
    the source says "no history of blood cough".
    """
    try:
        result = symptom_extractor.extract_symptoms(request.text)
        return SymptomExtractionResponse(
            request_id=str(uuid.uuid4()),
            symptoms=result["symptoms"],
            negated=result["negated"],
            matches=result["matches"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
