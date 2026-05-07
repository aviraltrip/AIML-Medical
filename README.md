---
title: PulsePoint AI
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# PulsePoint AI — Project Summary

A **FastAPI microservice** built for the MED-AI Hackathon 2025 that acts as the AI/ML "brain" for a family health network. The frontend (Next.js on Vercel) calls this service; the service itself is deployed on Hugging Face Spaces via Docker.

It exposes three engines under `/api/v1/{triage,predict,connect}` ([api/main.py](src/pulsepoint_ai/api/main.py)) and is built around a strong safety philosophy: **deterministic rules and ML are the floor, the LLM is grounded on top, and a hallucination guard blocks anything the LLM invents that can't be cross-checked**.

## The 3 Engines

### 1. Triage Engine — "Is this an emergency?"
Pipeline lives in [engines/triage/pipeline.py](src/pulsepoint_ai/engines/triage/pipeline.py). For a patient's symptoms + vitals, it runs a fixed 6-step chain:

1. **Vital Rules** ([safety/vital_rules.py](src/pulsepoint_ai/safety/vital_rules.py)) — deterministic YAML-driven rule engine. Reads thresholds from [configs/triage_rules.yaml](configs/triage_rules.yaml). Produces a "tier floor" (LOW → EMERGENCY) the LLM cannot override.
2. **LightGBM Triage Classifier** ([engines/triage/classifier/](src/pulsepoint_ai/engines/triage/classifier/)) — multiclass severity classifier trained with stratified k-fold CV ([train.py](src/pulsepoint_ai/engines/triage/classifier/train.py)), outputs probabilities + feature importance.
3. **RAG Retrieval** ([engines/triage/rag/retriever.py](src/pulsepoint_ai/engines/triage/rag/retriever.py)) — in-memory cosine similarity over a knowledge base ([data/kb/chunks.jsonl](data/kb/chunks.jsonl) + `embeddings.npz`) built from WHO/MOHFW guidelines in [data/raw/kb/](data/raw/kb/). Uses Gemini embeddings (`gemini-embedding-001`).
4. **LLM Reasoner** ([engines/triage/reasoner.py](src/pulsepoint_ai/engines/triage/reasoner.py)) — Gemini `gemini-flash-latest` (with 2.5/2.0/Llama 3.3/Gemma/Mistral fallbacks via LiteLLM, see [configs/models.yaml](configs/models.yaml)). Templated via [configs/prompts/reasoner_v1.j2](configs/prompts/reasoner_v1.j2), forced to JSON output.
5. **Hallucination Guard** ([safety/hallucination_guard.py](src/pulsepoint_ai/safety/hallucination_guard.py)) — set-difference check; blocks ICD-10 codes / red flags the LLM made up.
6. **Final tier = max(rule, classifier, reasoner)** — safety always wins.

The **Symptom Interviewer** ([engines/triage/interviewer.py](src/pulsepoint_ai/engines/triage/interviewer.py)) is a separate endpoint: a fine-tuned **FLAN-T5-small + LoRA** adapter ([models/interviewer_lora/](models/interviewer_lora/)) that generates the next clinically useful follow-up question. A Llama-3.1 LoRA training script also exists in [interview/train_lora.py](src/pulsepoint_ai/interview/train_lora.py). Falls back to rule-based questions when the adapter or torch isn't available.

### 2. Predict Engine — "What might it be?"
- **Disease Classifier** ([engines/predict/disease_classifier/infer.py](src/pulsepoint_ai/engines/predict/disease_classifier/infer.py)) — LightGBM, one-hot encoding of 130+ symptoms → top-5 ICD-10 codes with probabilities. Has a hardcoded rule-based fallback for when the model artifact is missing.
- **Lab Detector** ([engines/predict/lab_detector.py](src/pulsepoint_ai/engines/predict/lab_detector.py)) — pure regex + YAML reference ranges from [configs/lab_ranges.yaml](configs/lab_ranges.yaml), age/gender-aware. Acts as ground truth — the LLM cannot influence it.
- **Lab Explainer** ([engines/predict/lab_explainer.py](src/pulsepoint_ai/engines/predict/lab_explainer.py)) — Gemini generates plain-English explanations only for already-flagged abnormal labs; explanations are bound to the deterministic flag IDs and pass the hallucination guard. A medical disclaimer is appended.
- **Condition Cards** ([engines/predict/condition_cards.py](src/pulsepoint_ai/engines/predict/condition_cards.py)) — RAG-grounded patient-friendly summaries (Grade-6 reading level, 3 action steps, max 80 words).
- **Trends** ([engines/predict/trends.py](src/pulsepoint_ai/engines/predict/trends.py)) — linear regression for longitudinal vital tracking.

### 3. Connect Engine — "Hand off to a human"
- **MedReach** ([engines/connect/medreach.py](src/pulsepoint_ai/engines/connect/medreach.py)) — Gemini-powered <100-word doctor handoff summary, with a deterministic fallback paragraph if the LLM call fails.
- **Translation** ([engines/connect/translation.py](src/pulsepoint_ai/engines/connect/translation.py)) — local-language medical text translation.
- **Referral** ([engines/connect/referral.py](src/pulsepoint_ai/engines/connect/referral.py)) — doctor-ready PDF referral cards.

## AI/ML Inventory

| Component | Type | Where |
|---|---|---|
| Triage severity classifier | LightGBM multiclass (5 tiers) | [models/triage_lgbm/](models/triage_lgbm/) |
| Disease classifier | LightGBM 130+ symptoms → ICD-10 top-K | [models/disease_lgbm/](models/disease_lgbm/) |
| Symptom interviewer | FLAN-T5-small + LoRA adapter (peft) | [models/interviewer_lora/](models/interviewer_lora/) |
| RAG embeddings | `gemini-embedding-001` (3072-dim) | [data/kb/embeddings.npz](data/kb/embeddings.npz) |
| Reasoning LLM | Gemini Flash (primary) + LiteLLM fallback chain (Llama 3.3, Gemma 2, Mistral 7B via OpenRouter) | [llm/client.py](src/pulsepoint_ai/llm/client.py) |
| Vital rule engine | Deterministic YAML rules | [configs/triage_rules.yaml](configs/triage_rules.yaml) |
| Lab range detector | Regex + YAML reference ranges (40+ tests) | [configs/lab_ranges.yaml](configs/lab_ranges.yaml) |
| Trend analysis | scikit-learn linear regression | [engines/predict/trends.py](src/pulsepoint_ai/engines/predict/trends.py) |
| OCR / STT input | Tesseract + Whisper/AssemblyAI stubs | [input_layer/](src/pulsepoint_ai/input_layer/) |

## How a Triage Request Flows
1. Client POSTs symptoms + vitals to `/api/v1/triage/assess` ([api/routers/triage.py](src/pulsepoint_ai/api/routers/triage.py)).
2. `run_triage` runs the 6-step pipeline above in parallel-where-possible.
3. The response combines the ML probabilities, fired vital rules, top conditions, red flags, doctor briefing, citations, hallucination check status, and a recommended `next_action` from [safety/safety_rails.py](src/pulsepoint_ai/safety/safety_rails.py).

## Safety / Trust Layer
The recurring pattern is: **deterministic ground truth → LLM enrichment → set-difference guard → audit log**. The LLM never produces clinically actionable items (ICD-10 codes, lab flags, severity floor) without a rule-based or ML-based source it can be checked against ([safety/](src/pulsepoint_ai/safety/) — `vital_rules.py`, `hallucination_guard.py`, `safety_rails.py`, `audit_log.py`). All severity-tier merges take the *maximum* across rule/ML/LLM, so the LLM can only escalate, never downgrade.
