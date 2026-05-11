---
title: PulsePoint AI
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# PulsePoint — AI Family Health Network

[![Project Status: Active](https://img.shields.io/badge/Project%20Status-Active-brightgreen.svg)](https://github.com/aviraltrip/AIML-MedAI)
[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-blue.svg)](LICENSE)

PulsePoint is an AI-first medical engine for safe health triage, diagnostic guidance, and care coordination. It combines deterministic medical rules, LightGBM models, and grounded LLM reasoning to produce reliable clinical insights.

---

## Architecture

PulsePoint is built around three integrated engines:

1. **Triage Engine**
   - Applies deterministic vital sign rules and WHO/ESI thresholds.
   - Uses a LightGBM severity classifier (LOW / MEDIUM / HIGH / URGENT / EMERGENCY).
   - Runs a grounded LLM-based reasoner with retrieval over medical guidelines.
   - Includes a strict safety layer that blocks invented ICD-10 codes and other hallucinations.

2. **Predict Engine**
   - Predicts likely ICD-10 conditions from symptoms.
   - Generates patient-friendly condition summaries.
   - Detects abnormal lab results and explains them using reference ranges.
   - Extracts symptoms from raw clinical text with negation-aware logic.

3. **Connect Engine**
   - Summarizes AI findings for clinicians.
   - Generates referral-ready outputs.
   - Recommends nearby doctors using specialty matching, proximity, availability, and severity.

---

## Safety Principles

- Deterministic rules are the foundation.
- LLM outputs are grounded with retrieval and validation.
- Audit logs capture evidence for every decision.
- The system avoids creative medical output and only returns verified guidance.

---

## Tech Stack

- Python 3.11, FastAPI, Pydantic v2
- LightGBM classifiers for triage and disease prediction
- Gemini Flash for reasoning and interviewer prompts
- FLAN-T5-small + LoRA as a fallback interviewer model
- LiteLLM fallback chain for inference
- NumPy, scikit-learn, pandas for data and RAG support
- Tesseract OCR for lab report extraction

---

## Setup

### Prerequisites

- Python 3.11 or 3.12
- Tesseract OCR

### Install

```bash
git clone https://github.com/aviraltrip/AIML-MedAI.git
cd AIML-MedAI
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev,train]"
cp .env.example .env
```

Add your `GEMINI_API_KEY` and any required credentials to `.env`.

### Run

```bash
python -m pulsepoint_ai.engines.triage.app
```

---

## Project Layout

```text
src/pulsepoint_ai/
├── api/              # FastAPI endpoints
├── core/             # Config, schemas, shared utilities
├── engines/          # triage, predict, connect pipelines
├── input_layer/      # OCR / speech ingestion
├── interview/        # LoRA training/inference helpers
├── llm/              # LLM client abstractions
└── safety/           # rules, guards, logging
```

---

## Key Endpoints

- `POST /api/v1/triage/assess`
- `POST /api/v1/triage/interview`
- `POST /api/v1/predict/disease`
- `POST /api/v1/predict/labs`
- `POST /api/v1/predict/symptoms-from-text`
- `POST /api/v1/connect/care-locator`

---

## Notes

- Doctor ranking logic is configured in `configs/care_locator.yaml`.
- Evidence and rules are stored in YAML and config files, not hard-coded in Python.
- The service is designed as a modular API backend for a healthcare frontend.

---

