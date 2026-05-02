import json
from typing import Any

import litellm
import google.generativeai as genai
from pulsepoint_ai.core.config import get_models_config, get_settings

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
            import os
            os.environ["OPENROUTER_API_KEY"] = self.settings.openrouter_api_key

    async def complete_json(self, prompt: str, prompt_version: str = "v1") -> dict[str, Any]:
        """Completes a prompt and parses the response as JSON."""
        primary = self.config["primary"]
        
        if primary["provider"] == "gemini":
            try:
                # Use the full model path from the list
                model_name = primary["model"]
                if not model_name.startswith("models/"):
                    model_name = f"models/{model_name}"
                
                # Replace gemini-flash-latest with a known working identifier if needed
                if "flash-latest" in model_name:
                    model_name = "models/gemini-1.5-flash"

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
                return json.loads(content)
            except Exception as e:
                print(f"Native Gemini call failed: {e}")
                # Fallback to LiteLLM if native fails
                return await self._complete_with_litellm(prompt)
        else:
            return await self._complete_with_litellm(prompt)

    async def _complete_with_litellm(self, prompt: str) -> dict[str, Any]:
        primary = self.config["primary"]
        try:
            model_name = primary['model']
            if not model_name.startswith("models/"):
                model_name = f"models/{model_name}"
            
            # litellm expects gemini/ prefix
            litellm_model = model_name.replace("models/", "gemini/")

            response = await litellm.acompletion(
                model=litellm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2048,
            )
            content = response.choices[0].message.content
            
            # Clean up potential markdown formatting
            if content.startswith("```json"):
                content = content.replace("```json", "", 1).rsplit("```", 1)[0].strip()
            elif content.startswith("```"):
                content = content.replace("```", "", 1).rsplit("```", 1)[0].strip()
            
            return json.loads(content)
        except Exception as e:
            print(f"LiteLLM call failed: {e}")
            raise e
