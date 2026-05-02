import asyncio
import argparse
from pulsepoint_ai.engines.triage.pipeline import run_triage
from pulsepoint_ai.llm.client import LLMClient
from pulsepoint_ai.engines.triage.rag.retriever import Retriever
from pulsepoint_ai.core.schemas.triage import TriageAssessRequest
from pulsepoint_ai.core.schemas.common import PatientProfile, Vitals

async def chat():
    print("🏥 PulsePoint Medical Bot - AI Triage Assistant")
    print("Type 'exit' to quit.\n")
    
    llm = LLMClient()
    retriever = Retriever() # Assuming it's configured

    while True:
        symptoms_str = input("Describe your symptoms (e.g., 'fever, cough'): ")
        if symptoms_str.lower() == 'exit':
            break
            
        symptoms = [s.strip() for s in symptoms_str.split(",")]
        
        # Mocking vitals for the demo
        req = TriageAssessRequest(
            symptoms=symptoms,
            vitals=Vitals(heart_rate=80, bp_systolic=120, bp_diastolic=80, temperature=37.0),
            patient_profile=PatientProfile(age=30, gender="male")
        )
        
        print("\n🤔 Analyzing...")
        res = await run_triage(req, llm=llm, retriever=retriever)
        
        print(f"\n--- Assessment Results ---")
        print(f"SEVERITY: {res.severity.value.upper()}")
        print(f"RECOMMENDATION: {res.next_action}")
        print(f"\nREASONING: {res.doctor_briefing}")
        print(f"\nTOP CONDITIONS: {', '.join([c['name'] for c in res.top_conditions])}")
        print("-" * 30 + "\n")

if __name__ == "__main__":
    asyncio.run(chat())
