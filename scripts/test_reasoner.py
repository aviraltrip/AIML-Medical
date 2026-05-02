import os
import sys
import asyncio

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from pulsepoint_ai.engines.triage.reasoner import reason
from pulsepoint_ai.core.schemas.common import PatientProfile, Vitals, Gender
from pulsepoint_ai.llm.client import LLMClient

async def test_rag_reasoner():
    print("="*50)
    print("PULSEPOINT RAG REASONER TEST")
    print("="*50)

    # 1. Setup Mock Patient Data
    profile = PatientProfile(age=28, gender=Gender.FEMALE)
    vitals = Vitals(pulse_bpm=95, temp_c=38.2) # Slight fever
    symptoms = ["Severe pain around belly button", "pain moved to right lower stomach", "nausea"]

    # 2. Mock Retrieved Chunks
    retrieved_chunks = [
        {
            "id": "chunk_001",
            "text": "Appendicitis: Pain starting near the navel moving to the Right Lower Quadrant (RLQ). Pain increases with movement or coughing (McBurney's point tenderness).",
            "source": {
                "id": "common_conditions",
                "title": "NHS: Common Conditions",
                "url": "https://www.nhs.uk/conditions/appendicitis/"
            }
        },
        {
            "id": "chunk_002",
            "text": "ESI Level 2 (Emergent): High-risk situation, severe pain or distress. Assessment required within 10-15 minutes.",
            "source": {
                "id": "triage_guidelines",
                "title": "WHO/ESI Triage Standards",
                "url": "https://www.who.int/publications/i/item/9789241510219"
            }
        }
    ]

    print(f"Symptoms: {', '.join(symptoms)}")
    print("AI is cross-referencing with WHO/NHS guidelines...")

    # 3. Call Reasoner
    client = LLMClient()
    
    try:
        verdict = await reason(
            symptoms=symptoms,
            vitals=vitals,
            patient_profile=profile,
            rule_findings=[],
            classifier_output={"severity": "HIGH", "confidence": 0.85},
            retrieved_chunks=retrieved_chunks,
            llm=client
        )

        print("\n[AI REASONER VERDICT]:")
        print(f"Severity: {verdict.get('severity', 'UNKNOWN')}")
        print(f"Logic: {verdict.get('clinical_reasoning', verdict.get('reasoning_steps', ['No reasoning provided'])[0])}")
        print(f"Red Flags: {verdict.get('red_flags', [])}")
        print(f"Doctor Briefing: {verdict.get('doctor_briefing', 'N/A')}")
        
    except Exception as e:
        print(f"RAG Reasoner test failed: {e}")

    print("\n" + "="*50)

if __name__ == "__main__":
    asyncio.run(test_rag_reasoner())
