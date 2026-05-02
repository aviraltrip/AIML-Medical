# PulsePoint — AI Family Health Network (Blueprint v3.0)

This repository implements the modular **AI/ML Core** for the PulsePoint network, specifically designed for the **MED-AI Hackathon 2025**.

## 🏗️ The 3-Engine Architecture

### 1. Triage Engine (`engines/triage/`)
- **Vital Signs Guard:** Deterministic rules for immediate safety.
- **Symptom Interviewer:** Fine-tuned Llama 3.1 LoRA for clinical follow-up.
- **Triage Classifier:** LightGBM model for severity (Low/Medium/High/Critical).
- **RAG Reasoner:** Gemini 2.5 Flash grounded in WHO/MOHFW guidelines.

### 2. Predict Engine (`engines/predict/`)
- **Disease Classifier:** 130+ symptoms -> Top 5 ICD-10 conditions.
- **Condition Cards:** RAG-generated plain English health info.
- **Lab Analysis:** 40+ test detection + AI explanation.
- **Trend Intelligence:** Linear regression for health tracking.

### 3. Connect Engine (`engines/connect/`)
- **MedReach AI:** Async consult summarization & translation.
- **Referral logic:** PDF generation for doctor-ready cards.

## 🛡️ Ethical AI & Safety (`safety/`)
- **Hallucination Guard:** Deterministic grounding to prevent AI-invented labs/codes.
- **Bias Audit Log:** Continuous monitoring of model reliability.

## 🚀 Input Layer (`input_layer/`)
- **STT:** Online (AssemblyAI) & Offline (Whisper WASM).
- **OCR:** Medical report extraction (Tesseract + Sharp).

---
*Built for the KLE Technological University MED-AI Hackathon.*
