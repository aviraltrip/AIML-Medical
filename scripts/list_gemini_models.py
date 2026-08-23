import os
import sys

import google.generativeai as genai

sys.path.append(os.path.join(os.getcwd(), "src"))

from pulsepoint_ai.core.config import get_settings


def list_models():
    settings = get_settings()
    genai.configure(api_key=settings.google_api_key)

    print("Listing available Gemini models for your key:")
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"- {m.name}")
    except Exception as e:
        print(f"Error listing models: {e}")

if __name__ == "__main__":
    list_models()
