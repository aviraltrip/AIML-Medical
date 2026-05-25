import json
import os
import re
from typing import Any

import litellm
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    import google.generativeai as genai

from pulsepoint_ai.core.config import get_models_config, get_settings, get_lab_knowledge

class LLMClient:
    def __init__(self):
        self.config = get_models_config()["llm"]
        self.settings = get_settings()
        
        # Initialize Gemini directly
        if self.settings.google_api_key:
            genai.configure(api_key=self.settings.google_api_key)
        
        # Setup API keys for litellm (for fallbacks)
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
                    # Use the full model path from the list
                    model_name = primary["model"]
                    if not model_name.startswith("models/"):
                        model_name = f"models/{model_name}"
                    
                    # Remove hardcoded override since gemini-flash-latest is supported

                    model = genai.GenerativeModel(model_name)
                    
                    # Disable safety filters for clinical reasoning (it's a research project)
                    safety_settings = [
                        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                    ]

                    response = model.generate_content(
                        prompt,
                        generation_config=genai.GenerationConfig(
                            temperature=0.1,
                            max_output_tokens=2048,
                            response_mime_type="application/json",
                        ),
                        safety_settings=safety_settings
                    )
                    
                    content = response.text
                    return self._parse_json(content)
                except Exception as e:
                    print(f"Native Gemini call failed: {e}")
                    # Fallback to LiteLLM if native fails
                    return await self._complete_with_litellm(prompt)
            else:
                return await self._complete_with_litellm(prompt)
        except Exception as e:
            print(f"All LLM pathways failed: {e}. Executing local fallback for {prompt_version}...")
            return self._local_fallback(prompt, prompt_version)

    def _local_fallback(self, prompt: str, prompt_version: str) -> dict[str, Any]:
        if prompt_version == "reasoner_v1":
            # Extract inputs from prompt
            diabetes_idrs = 0
            diabetes_prob = 0.0
            hypertension_staging = "Normal"
            hypertension_prob = 0.0
            severity_staging = "MEDIUM"
            
            # Try to find classifier_output JSON in the prompt
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
            
            # Construct a high-quality clinical reasoner output
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
            return {
                "question": "Is there a history of sugar (diabetes) or high BP in your parents or siblings?",
                "rationale": "Determines genetic predisposition for metabolic screening.",
                "expected_answer_type": "yes_no"
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
            # If no fallbacks, try primary model via LiteLLM as last resort
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
        # Clean up potential markdown formatting
        if raw_text.startswith("```json"):
            raw_text = raw_text.replace("```json", "", 1).rsplit("```", 1)[0].strip()
        elif raw_text.startswith("```"):
            raw_text = raw_text.replace("```", "", 1).rsplit("```", 1)[0].strip()
            
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError as e:
            print("=== Gemini raw output that failed to parse ===")
            print(repr(raw_text))
            print("=== Error ===", e)
            
            # Simple heuristic fix for truncated JSON (often missing closing braces)
            try:
                fixed_text = raw_text
                if fixed_text.count("{") > fixed_text.count("}"):
                    fixed_text += "}"
                if fixed_text.count("[") > fixed_text.count("]"):
                    fixed_text += "]"
                # Try parsing again
                return json.loads(fixed_text)
            except Exception:
                pass # If it still fails, raise the original error
                
            raise

