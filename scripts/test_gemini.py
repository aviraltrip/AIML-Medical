import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("Error: GOOGLE_API_KEY not found in .env")
    exit(1)

genai.configure(api_key=api_key)

print("Attempting to list models and run a simple test...")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"Found model: {m.name}")
            
    # Try a simple generation with the most likely model
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content("Say hello!")
    print(f"Success! Response: {response.text}")
except Exception as e:
    print(f"Test failed: {e}")
