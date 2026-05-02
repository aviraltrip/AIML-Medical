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
│   └── connect/      # Summarization, referral, translation
├── input_layer/      # OCR (Tesseract) and STT (Whisper/AssemblyAI)
├── interview/        # Llama 3.1 LoRA training & inference logic
├── llm/              # Standardized LLM client (LiteLLM)
└── safety/           # Vital rules, hallucination guards, audit logs
```

---
*Built with ❤️ for the KLE Technological University MED-AI Hackathon.*
