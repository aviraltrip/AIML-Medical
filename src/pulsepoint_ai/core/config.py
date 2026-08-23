from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = REPO_ROOT / "configs"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)


    google_api_key: str | None = None
    openai_api_key: str | None = None
    openrouter_api_key: str | None = None
    hf_token: str | None = None


    configs_dir: Path = Field(default=CONFIGS_DIR)
    data_dir: Path = Field(default=REPO_ROOT / "data")
    models_dir: Path = Field(default=REPO_ROOT / "models")


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def get_models_config() -> dict[str, Any]:
    return _load_yaml(get_settings().configs_dir / "models.yaml")


@lru_cache(maxsize=1)
def get_lab_ranges() -> dict[str, Any]:
    return _load_yaml(get_settings().configs_dir / "lab_ranges.yaml")


@lru_cache(maxsize=1)
def get_lab_knowledge() -> dict[str, Any]:
    return _load_yaml(get_settings().configs_dir / "lab_knowledge.yaml")


@lru_cache(maxsize=1)
def get_symptoms() -> dict[str, Any]:
    return _load_yaml(get_settings().configs_dir / "symptoms.yaml")


@lru_cache(maxsize=1)
def get_rag_sources() -> dict[str, Any]:
    return _load_yaml(get_settings().configs_dir / "rag" / "sources.yaml")


@lru_cache(maxsize=1)
def get_triage_rules() -> dict[str, Any]:
    return _load_yaml(get_settings().configs_dir / "triage_rules.yaml")


@lru_cache(maxsize=1)
def get_care_locator_config() -> dict[str, Any]:
    return _load_yaml(get_settings().configs_dir / "care_locator.yaml")


@lru_cache(maxsize=1)
def get_doctors() -> list[dict[str, Any]]:
    path = get_settings().data_dir / "doctors.json"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, list) else []
