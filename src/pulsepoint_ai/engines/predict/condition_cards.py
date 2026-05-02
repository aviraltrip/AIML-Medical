"""Condition Cards: RAG-grounded patient education for predicted diseases."""
from __future__ import annotations

from typing import Any
from pulsepoint_ai.llm.client import LLMClient
from pulsepoint_ai.engines.triage.rag.retriever import Retriever

async def generate_condition_card(condition_name: str, *, llm: LLMClient, retriever: Retriever) -> dict[str, Any]:
    """Generates a plain-language education card for a specific condition."""
    
    # 1. Retrieve clinical context from KB
    query = f"What is {condition_name} and what are the first aid or action steps?"
    chunks = await retriever.search(query, top_k=2)
    context = "\n".join([c.text for c in chunks])

    # 2. Generate Card using LLM
    prompt = (
        f"Generate a medical condition card for '{condition_name}'.\n"
        f"Context from medical guidelines: {context}\n"
        "Instructions:\n"
        "1. Reading level: Grade 6.\n"
        "2. No jargon.\n"
        "3. Provide exactly 3 clear action steps.\n"
        "4. Max 80 words.\n"
        "Output JSON: {'summary': str, 'actions': [str, str, str]}"
    )
    
    card = await llm.complete_json(prompt, prompt_version="condition_card_v1")
    return card
