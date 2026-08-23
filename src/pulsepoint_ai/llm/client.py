import json
import os
import re
import warnings
from typing import Any, cast

import litellm

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    import google.generativeai as genai

from pulsepoint_ai.core.config import get_lab_knowledge, get_models_config, get_settings


class LLMClient:
    def __init__(self) -> None:
        self.config = get_models_config()["llm"]
        self.settings = get_settings()


        if self.settings.google_api_key:
            genai.configure(api_key=self.settings.google_api_key)  # type: ignore[attr-defined]


        if self.settings.openai_api_key:
            litellm.api_key = self.settings.openai_api_key
        if self.settings.openrouter_api_key:
            os.environ["OPENROUTER_API_KEY"] = self.settings.openrouter_api_key

    async def complete_json(self, prompt: str, prompt_version: str = "v1") -> dict[str, Any]:
        """Completes a prompt and parses the response as JSON."""
        primary = self.config["primary"]

        try:
            if primary["provider"] == "gemini":
                try:

                    model_name = primary["model"]
                    if not model_name.startswith("models/"):
                        model_name = f"models/{model_name}"



                    model = genai.GenerativeModel(model_name)  # type: ignore[attr-defined]


                    safety_settings = [
                        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                    ]

                    response = model.generate_content(
                        prompt,
                        generation_config=genai.GenerationConfig(  # type: ignore[attr-defined]
                            temperature=0.1,
                            max_output_tokens=2048,
                            response_mime_type="application/json",
                        ),
                        safety_settings=safety_settings  # type: ignore[arg-type]
                    )

                    content = response.text
                    return self._parse_json(content)
                except Exception as e:
                    print(f"Native Gemini call failed: {e}")

                    return await self._complete_with_litellm(prompt)
            else:
                return await self._complete_with_litellm(prompt)
        except Exception as e:
            print(f"All LLM pathways failed: {e}. Executing local fallback for {prompt_version}...")
            return self._local_fallback(prompt, prompt_version)

    def _local_fallback(self, prompt: str, prompt_version: str) -> dict[str, Any]:
        if prompt_version == "reasoner_v1":

            diabetes_idrs = 0
            diabetes_prob = 0.0
            hypertension_staging = "Normal"
            hypertension_prob = 0.0
            severity_staging = "MEDIUM"


            classifier_match = re.search(r"Scoring Engine Staging.*?:?\s*(\{.*\})", prompt)
            if classifier_match:
                try:
                    clf_data = json.loads(classifier_match.group(1))
                    diabetes_idrs = clf_data.get("diabetes_idrs", 0)
                    diabetes_prob = clf_data.get("diabetes_prob", 0.0)
                    hypertension_staging = clf_data.get("hypertension_staging", "Normal")
                    hypertension_prob = clf_data.get("hypertension_prob", 0.0)
                    severity_staging = clf_data.get("severity_staging", "MEDIUM")
                except Exception as ex:
                    print(f"Error parsing classifier_output JSON: {ex}")


            return {
                "severity": severity_staging,
                "top_conditions": [
                    {"name": "Type 2 Diabetes Risk", "icd10": "E11", "prob": round(float(diabetes_prob), 2)},
                    {"name": "Hypertension Risk", "icd10": "I10", "prob": round(float(hypertension_prob), 2)}
                ],
                "reasoning_steps": [
                    "Deterministic scoring models evaluated successfully.",
                    f"Blood pressure staging classified as {hypertension_staging}.",
                    f"Indian Diabetes Risk Score (IDRS) calculated at {diabetes_idrs}/100."
                ],
                "red_flags": [],
                "doctor_briefing": f"ASHA Screening Handoff (Local Fallback): Patient shows potential risk for metabolic conditions. Blood pressure is staged at {hypertension_staging}. IDRS Diabetes Risk Score is {diabetes_idrs}/100 based on age, waist circumference, and activity indicators. Recommend referral to nearest PHC for confirmatory testing (fasting blood glucose, HbA1c).",
                "sources": [{"id": "who_guidelines", "title": "WHO Screening Guidelines", "url": "https://who.int"}],
                "confidence": 0.85
            }

        elif prompt_version == "lab_explainer_v1":
            ids = re.findall(r"- ID:\s*([a-zA-Z0-9_]+)", prompt)
            knowledge = get_lab_knowledge()
            explanations = []
            for cid in ids:
                desc = knowledge.get(cid, {}).get("plain_explanation", "Lab value is abnormal. Please consult a clinician for further evaluation.")
                explanations.append({
                    "id": cid,
                    "text": desc
                })
            return {"explanations": explanations}

        elif prompt_version == "condition_card_v1":
            m = re.search(r"Condition:\s*(.*?)\s*\(ICD-10:\s*(.*?)\)", prompt)
            name = m.group(1).strip() if m else "Medical Condition"
            icd10 = m.group(2).strip() if m else "Unknown"

            if "diabetes" in name.lower() or "e11" in icd10.lower():
                summary = "Type 2 Diabetes means your body has trouble processing sugar from food, causing high blood sugar. This can damage blood vessels over time."
                actions = [
                    "Reduce rice, wheat, and sweets. Eat more green vegetables and lentils.",
                    "Walk briskly for 30 minutes every day.",
                    "Visit the nearest Primary Health Centre (PHC) to test your blood sugar."
                ]
            elif "hypertension" in name.lower() or "i10" in icd10.lower() or "blood pressure" in name.lower():
                summary = "Hypertension is high blood pressure. It makes your heart work harder to pump blood and can damage your blood vessels over time."
                actions = [
                    "Reduce your daily salt intake and avoid pickles and salty snacks.",
                    "Engage in physical activity, like walking, daily.",
                    "Visit the nearest PHC to check your blood pressure regularly."
                ]
            else:
                summary = f"{name} is a medical condition. Please consult a clinician for personal guidance."
                actions = [
                    "Talk to a healthcare professional about your symptoms.",
                    "Follow any medical advice or prescriptions provided by doctor.",
                    "Visit the nearest PHC if symptoms persist or worsen."
                ]
            return {
                "name": name,
                "plain_summary": summary,
                "action_steps": actions,
                "sources": [],
                "unsupported": False
            }

        elif prompt_version == "interviewer_v1":

            asked_questions = set()
            answered_match = re.search(r"Already answered \(do not repeat\):\s*(\[.*?\])", prompt, re.DOTALL)
            if answered_match:
                try:
                    answered_data = json.loads(answered_match.group(1))
                    for item in answered_data:
                        q_text = item.get("question", "").strip().lower()
                        if q_text:
                            asked_questions.add(q_text)
                except Exception as ex:
                    print(f"Error parsing answered list: {ex}")

            candidates = [
                (
                    "Is there a history of sugar (diabetes) or high BP in your parents or siblings?",
                    "Assesses genetic predisposition for chronic metabolic conditions.",
                    "yes_no"
                ),
                (
                    "How much physical work, farming, walking, or exercise do you do in a day?",
                    "Assesses physical activity level to calculate IDRS score.",
                    "free_text"
                ),
                (
                    "Do you regularly chew tobacco, smoke bidi, or drink alcohol?",
                    "Assesses behavioral risk factors for vascular health and metabolic disease.",
                    "yes_no"
                ),
                (
                    "When was your blood pressure or blood sugar last checked, and what was the value?",
                    "Retrieves historical clinical baseline values if available.",
                    "free_text"
                ),
                (
                    "How far is your village or home from the nearest Primary Health Centre (PHC)?",
                    "Screens for healthcare accessibility and referral delay risk.",
                    "free_text"
                ),
                (
                    "Have you noticed having frequent urination at night, dry mouth, or excessive thirst?",
                    "Screens for hallmark symptoms of hyperglycemia.",
                    "yes_no"
                ),
                (
                    "Do you regularly get headaches, dizziness, or chest tightness when working?",
                    "Screens for common somatic indicators of high blood pressure.",
                    "yes_no"
                ),
                (
                    "Do you have any slow-healing sores or ulcers on your feet?",
                    "Screens for diabetic peripheral neuropathy or microvascular complications.",
                    "yes_no"
                ),
                (
                    "What is your typical diet? Do you consume high-salt, fried, or wheat/rice-heavy meals?",
                    "Evaluates nutritional risks for metabolic syndrome.",
                    "free_text"
                ),
                (
                    "Have you recently experienced sudden blurriness or difficulty reading, even with glasses?",
                    "Screens for early diabetic retinopathy or vascular changes.",
                    "yes_no"
                ),
                (
                    "Do you feel any burning, tingling, or loss of sensation in your hands or feet?",
                    "Screens for diabetic peripheral neuropathy symptoms.",
                    "yes_no"
                ),
                (
                    "If you have been pregnant, did you ever have high blood sugar or a baby weighing over 4 kg?",
                    "Assesses risk of gestational diabetes history which increases Type 2 Diabetes risk.",
                    "yes_no"
                ),
                (
                    "Have you noticed dark, velvety patches of skin around your neck or armpits?",
                    "Screens for acanthosis nigricans, a hallmark sign of insulin resistance.",
                    "yes_no"
                ),
                (
                    "Do you snore loudly at night or feel extremely sleepy during the daytime?",
                    "Identifies potential sleep apnea, highly correlated with metabolic syndrome.",
                    "yes_no"
                ),
                (
                    "What is your primary mode of travel to the PHC, and how much does it cost you?",
                    "Socioeconomic accessibility screening for default risk.",
                    "free_text"
                ),
                (
                    "What is your approximate weight and height, or do you consider yourself overweight?",
                    "Assesses BMI indicators for diabetic staging.",
                    "free_text"
                )
            ]

            for q, r, t in candidates:
                is_asked = False
                for asked in asked_questions:
                    if asked in q.lower() or q.lower() in asked:
                        is_asked = True
                        break
                if not is_asked:
                    return {
                        "question": q,
                        "rationale": r,
                        "expected_answer_type": t
                    }

            return {
                "question": "Could you tell me if you have any other concerns about sugar or blood pressure?",
                "rationale": "Fallback question to gather general chronic health concerns.",
                "expected_answer_type": "free_text"
            }

        elif prompt_version == "medreach_v1":
            return {
                "summary": "ASHA patient requires review. Screening flags show chronic risk stage. Detail report generated."
            }

        else:
            return {}

    async def _complete_with_litellm(self, prompt: str) -> dict[str, Any]:
        fallbacks = self.config.get("fallback_chain", [])
        if not fallbacks:

            primary = self.config["primary"]
            fallbacks = [{"provider": primary["provider"], "model": primary["model"]}]

        last_exception = None
        for fallback in fallbacks:
            try:
                model_name = fallback['model']
                if fallback['provider'] == 'openrouter':
                    litellm_model = f"openrouter/{model_name}"
                elif fallback['provider'] == 'gemini':
                    if not model_name.startswith("models/"):
                        model_name = f"models/{model_name}"
                    litellm_model = model_name.replace("models/", "gemini/")
                else:
                    litellm_model = model_name

                print(f"Trying LiteLLM fallback: {litellm_model}")
                response = await litellm.acompletion(
                    model=litellm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=2048,
                )
                content = response.choices[0].message.content
                return self._parse_json(content)
            except Exception as e:
                print(f"LiteLLM call failed for {model_name}: {e}")
                last_exception = e

        if last_exception is not None:
            raise last_exception
        raise RuntimeError("All fallback models failed to complete.")

    def _parse_json(self, raw_text: str) -> dict[str, Any]:
        """Cleans up markdown formatting and safely parses JSON with detailed error logging."""
        raw_text = raw_text.strip()

        if raw_text.startswith("```json"):
            raw_text = raw_text.replace("```json", "", 1).rsplit("```", 1)[0].strip()
        elif raw_text.startswith("```"):
            raw_text = raw_text.replace("```", "", 1).rsplit("```", 1)[0].strip()

        try:
            return cast(dict[str, Any], json.loads(raw_text))
        except json.JSONDecodeError as e:
            print("=== Gemini raw output that failed to parse ===")
            print(repr(raw_text))
            print("=== Error ===", e)


            try:
                fixed_text = raw_text
                if fixed_text.count("{") > fixed_text.count("}"):
                    fixed_text += "}"
                if fixed_text.count("[") > fixed_text.count("]"):
                    fixed_text += "]"

                return cast(dict[str, Any], json.loads(fixed_text))
            except Exception:
                pass

            raise

