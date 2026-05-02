import os
from pathlib import Path
from typing import Optional

import httpx
from pulsepoint_ai.core.config import get_settings

class STTEngine:
    """PulsePoint Speech-to-Text Engine (AssemblyAI + Whisper)."""
    
    def __init__(self):
        self.settings = get_settings()
        self.assembly_api_key = self.settings.assemblyai_api_key

    async def transcribe_online(self, audio_path: Path) -> str:
        """Transcribes audio using AssemblyAI (Hindi/English auto-detect)."""
        if not self.assembly_api_key:
            return "Error: AssemblyAI API Key not configured."
            
        # Implementation of AssemblyAI async upload & transcription
        # For a hackathon, we use their fast endpoint or the SDK
        headers = {"authorization": self.assembly_api_key}
        
        async with httpx.AsyncClient() as client:
            # 1. Upload
            with open(audio_path, "rb") as f:
                upload_response = await client.post(
                    "https://api.assemblyai.com/v2/upload",
                    headers=headers,
                    content=f
                )
            upload_url = upload_response.json()["upload_url"]
            
            # 2. Transcribe
            transcript_response = await client.post(
                "https://api.assemblyai.com/v2/transcript",
                headers=headers,
                json={
                    "audio_url": upload_url,
                    "language_detection": True
                }
            )
            transcript_id = transcript_response.json()["id"]
            
            # 3. Polling (simplified for hackathon logic)
            while True:
                polling_response = await client.get(
                    f"https://api.assemblyai.com/v2/transcript/{transcript_id}",
                    headers=headers
                )
                status = polling_response.json()["status"]
                if status == "completed":
                    return polling_response.json()["text"]
                elif status == "error":
                    return "Error: Transcription failed."
                # await asyncio.sleep(1) # In a real system

    async def transcribe_offline(self, audio_path: Path) -> str:
        """Fallback Whisper transcription for Rural Mode (2G/Offline)."""
        # In a real build, we'd use 'openai-whisper' or 'faster-whisper'
        # For now, we provide the logic block for integration
        try:
            import whisper
            model = whisper.load_model("tiny") # Blueprint specifies tiny.en for 2G
            result = model.transcribe(str(audio_path))
            return result["text"]
        except ImportError:
            return "Error: Whisper (openai-whisper) not installed."

    async def process_audio(self, audio_path: str, offline: bool = False) -> str:
        """Routes transcription based on network status (Rural Mode)."""
        path = Path(audio_path)
        if offline:
            return await self.transcribe_offline(path)
        return await self.transcribe_online(path)

# Global instance
stt_engine = STTEngine()
