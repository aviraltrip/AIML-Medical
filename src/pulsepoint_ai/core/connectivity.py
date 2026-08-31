"""Rural Mode Utility: Handles adaptive AI routing for low-connectivity (2G)."""
from __future__ import annotations

from enum import Enum


class NetworkTier(str, Enum):
    BROADBAND = "broadband"
    RURAL = "rural"
    OFFLINE = "offline"

class AIRouter:
    """Smart router to switch between cloud LLMs and local lightweight models."""

    def __init__(self, current_tier: NetworkTier = NetworkTier.BROADBAND):
        self.tier = current_tier

    def get_stt_mode(self) -> str:
        """Decides whether to use AssemblyAI (online) or Whisper (offline)."""
        if self.tier in {NetworkTier.RURAL, NetworkTier.OFFLINE}:
            return "offline_whisper_wasm"
        return "online_assembly_ai"

    def get_llm_mode(self) -> str:
        """Decides whether to use Gemini (online) or a cached local response."""
        if self.tier == NetworkTier.OFFLINE:
            return "local_cache_only"
        elif self.tier == NetworkTier.RURAL:
            return "compressed_llm_payload"
        return "full_gemini_flash"

    def update_tier_from_latency(self, latency_ms: int):
        """Automatically updates the tier based on real-time latency checks."""
        if latency_ms > 2000:
            self.tier = NetworkTier.RURAL
        elif latency_ms < 0:
            self.tier = NetworkTier.OFFLINE
        else:
            self.tier = NetworkTier.BROADBAND
