"""Prompt rendering logic using Jinja2 templates."""
from __future__ import annotations

import jinja2

from pulsepoint_ai.core.config import get_settings


def render_prompt(template_name: str, **kwargs) -> str:
    """Renders a prompt template from configs/prompts/."""
    settings = get_settings()
    template_dir = settings.configs_dir / "prompts"

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(template_dir)),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True
    )

    try:
        template = env.get_template(f"{template_name}.j2")
        return template.render(**kwargs)
    except jinja2.TemplateNotFound:
        raise FileNotFoundError(f"Prompt template {template_name}.j2 not found in {template_dir}") from None
