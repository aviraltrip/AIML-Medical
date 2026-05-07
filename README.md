---
title: PulsePoint AI
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---
# PulsePoint — AI Family Health Network (v3.0) 🏥

[![Project Status: Active](https://img.shields.io/badge/Project%20Status-Active-brightgreen.svg)](https://github.com/aviraltrip/AIML-MedAI)
[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-blue.svg)](LICENSE)
[![Built for: MED-AI Hackathon 2025](https://img.shields.io/badge/Hackathon-MED--AI%202025-orange.svg)](https://kletech.ac.in)

**PulsePoint** is a state-of-the-art AI/ML medical engine designed to provide reliable, safety-first health triage and diagnostics. Built for the **MED-AI Hackathon 2025**, it bridges the gap between raw medical data and actionable clinical insights using a multi-engine architecture grounded in deterministic medical rules and advanced Large Language Models (LLMs).

---

## 🏗️ The 3-Engine Architecture

### 1. 🛡️ Triage Engine
The core safety layer of PulsePoint, ensuring every patient receives the right level of care.
- **Vital Signs Guard:** A deterministic rule engine that evaluates vitals (HR, BP, Temp) against clinical thresholds for immediate "Red Flag" detection.
- **Symptom Interviewer:** A fine-tuned **Llama 3.1 8B (LoRA)** model that conducts structured medical interviews to extract nuanced clinical details.
- **Triage Classifier:** A **LightGBM** model trained on clinical datasets to categorize severity (Low, Medium, High, Critical) with SHAP-based feature importance.
- **RAG Reasoner:** A **Gemini 2.5 Flash** agent grounded in WHO and MOHFW guidelines via Vector Search (RAG) to provide evidence-based triage justifications.

### 2. 🔍 Predict Engine
Transforming data into foresight with advanced diagnostic modules.
- **Disease Classifier:** Maps 130+ symptoms to the top 5 most likely **ICD-10** conditions using a hybrid ML approach.
- **Condition Cards:** RAG-generated, plain-English summaries of medical conditions to empower patients with knowledge.
- **Lab Intelligence:** 
  - **Lab Detector:** High-precision OCR extraction + Regex matching for 40+ common lab tests.
  - **Lab Explainer:** LLM-powered interpretation of abnormal results, grounded in canonical reference ranges.
- **Trend Intelligence:** Linear regression models for longitudinal health tracking (e.g., glucose or blood pressure trends).

### 3. 🤝 Connect Engine
Seamlessly connecting AI insights to human clinical workflows.
- **MedReach AI:** Automated, asynchronous summarization of patient-AI interactions for doctor review.
- **Smart Referrals:** Generation of doctor-ready PDF referral cards with structured clinical briefings.
- **Language Bridge:** Real-time translation of medical summaries into local languages to ensure healthcare accessibility.
- **HyperLocal Care Matching:** AI-ranked nearby doctor recommendations driven directly by upstream pipeline outputs (ICD-10 codes from the Predict Engine + severity tier from the Triage Engine). Four-stage chain:
  - **Multi-label Specialty Mapper** — longest-prefix ICD-10 → specialty classifier with deterministic generalist fallback (e.g., `I21.9` → Cardiologist, `J45.909` + `E11.9` → Pulmonologist + Endocrinologist).
  - **Haversine Geo Filter** — great-circle distance treated as a continuous feature, not a hard cutoff (same pattern Swiggy/Uber use).
  - **Weighted Relevance Scorer** — learning-to-rank style linear combination: `0.4·specialty_match + 0.3·proximity + 0.2·availability + 0.1·rating`. Every weight is in YAML, not Python.
  - **Urgency-Conditioned Sort** — EMERGENCY/URGENT cases force same-day-only filtering and sort by raw distance; routine cases sort by score. The triage engine's output literally changes ranking behavior at runtime.

---

## 🛡️ Safety & Ethical AI
We prioritize patient safety over model "creativity."
- **Hallucination Guard:** A deterministic validation layer that blocks AI-generated ICD-10 codes or lab names that aren't present in our verified knowledge base.
- **Deterministic Grounding:** All AI outputs are cross-referenced against high-fidelity medical rules.
- **Audit Logging:** Every decision made by the engines is logged with its supporting evidence (vitals, retrieved guidelines, classifier confidence).

---

## 🚀 Technical Stack
- **Core:** Python 3.11, FastAPI, Pydantic v2
- **Models:** Llama 3.1 (Fine-tuned), Gemini 1.5/2.0 Flash, LightGBM
- **Inference:** LiteLLM, OpenAI/Google SDKs
- **Data & RAG:** ChromaDB (Vector DB), Scikit-learn, Pandas
- **Ops:** Redis (Caching), Celery (Async Tasks), Prometheus/OpenTelemetry

---

## 🛠️ Getting Started

### Prerequisites
- Python 3.11 or 3.12
- Tesseract OCR (for medical report extraction)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/aviraltrip/AIML-MedAI.git
   cd AIML-MedAI
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -e ".[dev,train]"
   ```
4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Add your GEMINI_API_KEY and other credentials to .env
   ```

### Running the Demo
Experience the Triage Engine in action via the interactive CLI:
```bash
python -m pulsepoint_ai.engines.triage.app
```

---

## 📂 Project Structure
```text
src/pulsepoint_ai/
├── api/              # FastAPI routers and endpoints
├── core/             # Configuration, schemas, and shared logic
├── engines/          # The 3-Engine Core
│   ├── triage/       # Pipeline, classifier, RAG reasoner
│   ├── predict/      # Disease classification, labs, trends
│   └── connect/      # Summarization, referral, translation, care_locator
├── input_layer/      # OCR (Tesseract) and STT (Whisper/AssemblyAI)
├── interview/        # Llama 3.1 LoRA training & inference logic
├── llm/              # Standardized LLM client (LiteLLM)
└── safety/           # Vital rules, hallucination guards, audit logs
```

---

## 🌐 Developer Quick-Start (Full-Stack Integration)

This engine is live at: `https://aviraltrip-pulsepoint-ai.hf.space`
Interactive Documentation: `https://aviraltrip-pulsepoint-ai.hf.space/docs`

### 1. The Triage Flow (Main Feature)
Call this when a user submits their symptoms and vitals. It returns the severity tier and doctor reasoning.

**Endpoint:** `POST /api/v1/triage/assess`

```javascript
const triageResponse = await fetch('https://aviraltrip-pulsepoint-ai.hf.space/api/v1/triage/assess', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    patient_id: "USER_123",
    symptoms: ["fever", "cough", "shortness of breath"],
    vitals: {
      heart_rate: 95,
      bp_systolic: 120,
      bp_diastolic: 80,
      temperature: 38.5
    },
    patient_profile: { age: 25, gender: "female" }
  })
});

const result = await triageResponse.json();
console.log(result.severity); // "MEDIUM"
console.log(result.doctor_briefing); // Clinical reasoning for the UI
```

### 2. The Symptom Interviewer (Interactive Chat)
Call this to get the next "smart" follow-up question during a symptom check.

**Endpoint:** `POST /api/v1/triage/interview`

```javascript
const interviewResponse = await fetch('https://aviraltrip-pulsepoint-ai.hf.space/api/v1/triage/interview', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    symptoms: ["headache", "nausea"],
    patient_profile: { age: 45, gender: "male" }
  })
});

const data = await interviewResponse.json();
console.log(data.question); // "Is the headache throbbing or sharp?"
```

### 3. Lab Intelligence (OCR Interpretation)
Call this when a user uploads a medical report (pass the raw OCR text).

**Endpoint:** `POST /api/v1/predict/labs`

```javascript
const labResponse = await fetch('https://aviraltrip-pulsepoint-ai.hf.space/api/v1/predict/labs', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    ocr_text: "Hemoglobin: 10.2 g/dL, WBC: 4500...",
    age: 30,
    gender: "female"
  })
});

const labResult = await labResponse.json();
// Returns an array of "flags" with abnormal values and AI explanations
```

### 4. HyperLocal Care Matching (Doctor Recommendations)
Call this **after** `/triage/assess` and `/predict/disease` — feed the ICD-10 codes and severity tier from those responses, plus the patient's coordinates from `navigator.geolocation`. Returns a ranked list of nearby doctors with everything needed to render map markers and contact cards.

**Endpoint:** `POST /api/v1/connect/care-locator`

```javascript
// triageResult and diseaseResult come from the prior two calls.
const careResponse = await fetch('https://aviraltrip-pulsepoint-ai.hf.space/api/v1/connect/care-locator', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    icd10_codes: diseaseResult.predictions.map(p => p.icd10),  // from /predict/disease
    severity_tier: triageResult.severity,                       // from /triage/assess
    patient_lat: 15.3650,                                       // navigator.geolocation
    patient_lon: 75.1240,
    radius_km: 25                                               // optional UI slider
  })
});

const { doctors, required_specialties, sort_policy } = await careResponse.json();
// doctors[0] = { name, specialty, clinic, phone, lat, lon, distance_km, score,
//                available_today, rating, languages, fee_inr, score_breakdown }
// Drop lat/lon straight into a Mapbox or Google Maps component.
```

**Why this is AI/ML, not just a geo filter:**
- *Multi-label classification* — multiple ICD-10 codes resolve to multiple specialties simultaneously.
- *Learning-to-rank* — weighted feature vector across specialty match, proximity, availability, and rating.
- *Context-aware inference* — the upstream severity tier reshapes the ranking logic at runtime (EMERGENCY ⇒ same-day filter + distance sort).
- *Geospatial features* — haversine distance fed into the scorer as a continuous feature.
- *Deterministic fallback* — same architectural pattern as the rest of PulsePoint: rules catch what the model misses.

All ranking behavior lives in [configs/care_locator.yaml](configs/care_locator.yaml); the (mock) doctor catalog is [data/doctors.json](data/doctors.json) — swap for a Postgres query without touching the API.

---

## 🚀 Deployment Summary
- **Frontend**: Host on **Vercel** (Next.js).
- **AI Brain**: Hosted on **Hugging Face Spaces** (Docker).
- **Environment Variables**: Make sure to add `GEMINI_API_KEY` to your Hugging face Space settings.

---
*Built with ❤️ for the KLE Technological University MED-AI Hackathon.*
