---
title: PulsePoint AI - Rural Metabolic Early Screening
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# PulsePoint — Rural Diabetes & Hypertension Early Screening Platform

[![Project Status: Active](https://img.shields.io/badge/Project%20Status-Active-brightgreen.svg)](https://github.com/aviraltrip/AIML-MedAI)
[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-blue.svg)](LICENSE)

PulsePoint is a production-grade, clinical reasoning backend tailored for ASHA and ANM community health workers to conduct early screening of Type 2 Diabetes, Hypertension, and Metabolic Syndrome in underserved rural Indian populations. It integrates deterministic clinical scoring, a LightGBM default model, and LLM-grounded decision support to deliver robust, error-tolerant triage.

---

## Key Features

1. **Deterministic Metabolic Scoring (MDRF & JNC-8/WHO)**
   * **Indian Diabetes Risk Score (IDRS)**: Calculates points based on age, waist circumference, physical activity, and family history, with keywords-based fallback logic.
   * **Blood Pressure Staging**: Stage 1/2 Hypertension and Hypertensive Crisis thresholds based on JNC-8 guidelines.
   * **Chronic-to-Triage Severity Mapping**: Safely maps metabolic and vital-sign risks to ESI-compatible tiers (`LOW` to `EMERGENCY`).

2. **LightGBM Referral Adherence (Loss-to-Follow-Up) Predictor**
   * Employs a LightGBM binary classifier to predict the likelihood of a patient defaulting on their confirmatory Primary Health Centre (PHC) visit.
   * Leverages rural context signals: village-to-PHC travel distance, loss of daily wages (laborer/farming occupations), clinical severity, and adherence history.
   * Automatically triggers high-priority alerts (`[ALERT: High default risk - ASHA home visit required]`) in the `next_action` field if the probability exceeds `60%`.

3. **Fuzzy OCR Lab Report Parser**
   * Robust regex logic with spelling, space, and hyphen tolerances tailored for low-resource scanning noise.
   * Parses metabolic markers: Fasting Blood Sugar (FBS), Postprandial Blood Sugar (PPBS), HbA1c, microalbumin, and urine protein.
   * Enforces diagnostic thresholds: FBS (>=100/126 mg/dL), HbA1c (>=5.7%/6.5%), microalbuminuria (>=30 mg/L), and proteinuria (>=20 mg/dL).

4. **Dynamic Local LLM Fallback (Zero-Downtime Demo)**
   * Safe and stable execution even without API keys or under rate-limits. If the Google Gemini/LiteLLM call fails, the client automatically executes local deterministic reasoning steps, returning valid clinical briefings, condition cards, and translations without raising HTTP 500 errors.

---

## Tech Stack

* **Core**: Python 3.11, FastAPI, Pydantic v2
* **ML Classifiers**: LightGBM for referral adherence and disease risk staging
* **LLM Engine**: Gemini Flash for reasoning summaries and interviewer questions
* **Offline Fallbacks**: FLAN-T5-small + LoRA adapter CPU inference, rule-based fallbacks
* **Core Libraries**: NumPy, scikit-learn, pandas, Jinja2
* **Lab Ingestion**: Tesseract OCR for text extraction

---

## Setup & Installation

### Prerequisites
* Python 3.11 or 3.12
* Tesseract OCR installed on the system path

### Install Dependencies
```bash
git clone https://github.com/aviraltrip/AIML-MedAI.git
cd AIML-MedAI
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev,train]"
```

Configure your environment variables in `.env` (copying from `.env.example`).

---

## Running the Application

### 1. Start the FastAPI Server
To launch the backend server locally on port 7860 (exposed for Hugging Face Spaces):
```bash
python main.py
```

### 2. Run Verification Tests
To run the chronic refactor test suite covering calculations, parsing, ML classifiers, and API contracts:
```bash
# Windows
$env:PYTHONPATH=".;src"; .venv\Scripts\python.exe tests/run_tests.py

# Linux / macOS
PYTHONPATH=.:src .venv/bin/python tests/run_tests.py
```

### 3. Re-Train Adherence Model
To re-fit the LightGBM Loss-to-Follow-Up model:
```bash
python scripts/train_adherence.py
```

---

## Project Layout

```text
src/pulsepoint_ai/
├── api/              # FastAPI routers, endpoints, and CORS middleware
├── core/             # Reference ranges, settings, and Pydantic schemas
├── engines/          # Core analytics engine pipelines
│   ├── connect/      # Care locator, MedReach summaries, and translations
│   ├── predict/      # OCR lab detectors, condition cards, and symptom extractors
│   └── triage/       # Interviewers, reasoner prompts, and RAG retrievers
├── llm/              # LLM client setup, LiteLLM fallbacks, and local mock fallback
└── safety/           # Hardcoded vital rules and safety next-action rails
```
