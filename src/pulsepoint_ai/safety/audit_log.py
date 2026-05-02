"""PulsePoint Bias Audit Log - Tracks AI safety and hallucination metrics."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from pulsepoint_ai.core.config import get_settings

class AuditLogger:
    def __init__(self):
        self.settings = get_settings()
        self.log_dir = self.settings.data_dir / "audit"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "bias_audit.jsonl"

    def log_interaction(
        self,
        request_id: str,
        model_name: str,
        prompt_version: str,
        input_summary: dict[str, Any],
        output_verdict: str,
        hallucination_detected: bool,
        rag_sources: list[str] = None
    ):
        """Logs a single AI interaction for the audit trail."""
        entry = {
            "timestamp": time.time(),
            "request_id": request_id,
            "model": model_name,
            "prompt_version": prompt_version,
            "inputs": input_summary,
            "verdict": output_verdict,
            "hallucination": hallucination_detected,
            "rag_sources": rag_sources or [],
            "status": "flagged" if hallucination_detected else "verified"
        }
        
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
            
        print(f"📄 Audit Log: Request {request_id} recorded ({entry['status']})")

    def export_csv(self):
        """Generates a CSV for judges to review model performance."""
        # This would convert the jsonl to a readable CSV
        pass

# Global instance
audit_logger = AuditLogger()
