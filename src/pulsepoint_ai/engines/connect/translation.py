"""Translation Engine for doctor-patient communication (English -> Hindi/Marathi)."""
from __future__ import annotations

import httpx
from pulsepoint_ai.core.config import get_settings

class TranslationEngine:
    def __init__(self):
        self.base_url = "https://translate.googleapis.com/translate_a/single"

    async def translate(self, text: str, target_lang: str = "kn") -> str:
        """Translates text using the Google Translate proxy."""
        if not text:
            return ""
            
        params = {
            "client": "gtx",
            "sl": "en",
            "tl": target_lang,
            "dt": "t",
            "q": text,
        }
        
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(self.base_url, params=params)
                resp.raise_for_status()
                # Google returns a complex nested list: [[["translated", "original", ...]]]
                data = resp.json()
                translated = "".join([part[0] for part in data[0]])
                return translated
        except Exception as e:
            print(f"Translation failed: {e}")
            return text # Fallback to original

# Global instance
translator = TranslationEngine()


async def translate_medical_text(text: str, target_language: str, *, llm=None) -> str:
    """Router-facing entry point. `llm` is accepted for API compatibility but unused
    (the Google Translate gtx proxy is sufficient and free for short medical strings).
    """
    return await translator.translate(text, target_lang=target_language)
