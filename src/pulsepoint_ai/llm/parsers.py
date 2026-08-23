import json
import re
from typing import Any, cast


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```json"):
        text = text.replace("```json", "", 1).rsplit("```", 1)[0].strip()
    elif text.startswith("```"):
        text = text.replace("```", "", 1).rsplit("```", 1)[0].strip()

    try:
        return cast(dict[str, Any], json.loads(text))
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if match:
            try:
                return cast(dict[str, Any], json.loads(match.group(1)))
            except json.JSONDecodeError:
                pass
        raise
