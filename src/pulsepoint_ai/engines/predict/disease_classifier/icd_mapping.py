"""ICD-10 code → human-readable name. Loaded from a static JSON dictionary
that lives in data/icd10_dictionary.json (built by scripts/build_dataset.py)."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_DICT_PATH = Path("data/icd10_dictionary.json")


@lru_cache(maxsize=1)
def _load_dict() -> dict[str, str]:
    if _DICT_PATH.exists():
        return json.loads(_DICT_PATH.read_text(encoding="utf-8"))
    return {}


def name_for_icd10(code: str) -> str:
    return _load_dict().get(code, code)
