from fastapi import APIRouter, HTTPException, Depends
from pulsepoint_ai.core.schemas.triage import TriageAssessRequest, TriageAssessResponse
from pulsepoint_ai.core.schemas.interview import InterviewerRequest, InterviewerResponse
from pulsepoint_ai.engines.triage.pipeline import run_triage
from pulsepoint_ai.engines.triage.interviewer import next_question
from pulsepoint_ai.llm.client import LLMClient
from pulsepoint_ai.engines.triage.rag.retriever import Retriever
import uuid

router = APIRouter()

# Initialize shared components
llm_client = LLMClient()
retriever = Retriever()

@router.post("/assess", response_model=TriageAssessResponse)
async def assess_triage(request: TriageAssessRequest):
    """
    Perform a complete medical triage assessment using the 3-Engine pipeline.
    Includes vital rule evaluation, ML classification, and RAG-grounded LLM reasoning.
    """
    try:
        response = await run_triage(request, llm=llm_client, retriever=retriever)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/interview", response_model=InterviewerResponse)
async def conduct_interview(request: InterviewerRequest):
    """
    Generate the next clinical question based on the patient's symptoms and previous answers.
    Uses a fine-tuned Llama 3.1 model.
    """
    try:
        response = await next_question(request, llm=llm_client)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
