"""
Chat Engine — Llama-based Conversational AI with strict data grounding.

All answers are generated ONLY from real machine data injected into
the system prompt. The LLM is explicitly instructed never to
hallucinate or provide generic explanations.

Supports two providers:
  • Groq (cloud API) — fastest, uses llama-3.3-70b-versatile
  • Ollama (local)   — fully offline, uses llama3

Design decisions:
  • System prompt is rebuilt for EVERY request with fresh device
    context so the LLM always has current data.
  • The context_builder module provides the structured JSON;
    this module handles prompt assembly and LLM communication.
  • Uses httpx for async HTTP calls to both providers.
  • Conversation history is NOT maintained server-side — each
    request is self-contained with full context injection.
    This prevents stale data from polluting answers.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

import httpx

import config
import context_builder
from device_registry import DeviceRegistry

logger = logging.getLogger(__name__)

# ─── System Prompt Template ─────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """You are an expert industrial IoT maintenance assistant for a predictive maintenance platform.

STRICT RULES — YOU MUST FOLLOW THESE WITHOUT EXCEPTION:
1. Answer ONLY using the provided machine data below. Do NOT guess or make up information.
2. Do NOT provide generic maintenance advice that isn't grounded in the specific data.
3. If the data shows the machine is healthy, say so. Do NOT invent problems.
4. Always reference specific sensor values, thresholds, and trends from the data.
5. If you don't have enough data to answer, say "Insufficient data to determine this."
6. Use precise numbers from the data — never approximate or round differently.
7. Format your responses clearly with bullet points for multiple findings.

CURRENT MACHINE DATA:
```json
{device_context}
```

OPERATING THRESHOLDS REFERENCE:
- Vibration HIGH: >{vib_threshold}g
- Temperature HIGH: >{temp_threshold}°C
- Current normal range: {current_range}A

Answer the user's question using ONLY the data above."""


def _build_system_prompt(device_context: Dict[str, Any]) -> str:
    """Build the system prompt with injected device context."""
    thresholds = device_context.get("operating_thresholds", {})
    return SYSTEM_PROMPT_TEMPLATE.format(
        device_context=json.dumps(device_context, indent=2, default=str),
        vib_threshold=thresholds.get("vibration_high_g", "N/A"),
        temp_threshold=thresholds.get("temperature_high_celsius", "N/A"),
        current_range=thresholds.get("current_normal_range_amps", "N/A"),
    )


# ─── Main Chat Function ─────────────────────────────────────────

async def chat(
    device_id: str,
    user_message: str,
    registry: DeviceRegistry,
    question_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process a chat request for a specific device.

    Args:
        device_id: The device to query about
        user_message: The user's question
        registry: Device registry for context lookup
        question_id: Optional structured question ID
            (WHY_MAINTENANCE, IS_MACHINE_SAFE, etc.)

    Returns:
        {
            "response": str,
            "device_id": str,
            "question_id": str or None,
            "context_used": dict,
            "provider": str
        }
    """
    # 1. Build fresh device context
    device_context = context_builder.build_device_context(
        device_id, registry,
    )

    # 2. Resolve question (template or raw)
    resolved_question = context_builder.resolve_question(
        question_id, user_message, device_id,
    )

    # 3. Build system prompt with data injection
    system_prompt = _build_system_prompt(device_context)

    # 4. Call LLM
    provider = config.LLM_PROVIDER.lower()
    if provider == "groq":
        response_text = await _call_groq(system_prompt, resolved_question)
    elif provider == "ollama":
        response_text = await _call_ollama(system_prompt, resolved_question)
    else:
        response_text = (
            f"Unknown LLM provider: {provider}. "
            f"Set LLM_PROVIDER to 'groq' or 'ollama'."
        )

    return {
        "response": response_text,
        "device_id": device_id,
        "question_id": question_id,
        "context_used": {
            "status_summary": device_context.get("status_summary", ""),
            "prediction": device_context.get("prediction", {}),
        },
        "provider": provider,
    }


# ─── Groq Provider ──────────────────────────────────────────────

async def _call_groq(
    system_prompt: str,
    user_message: str,
) -> str:
    """Call the Groq API (OpenAI-compatible) with Llama 3.x."""
    if not config.GROQ_API_KEY:
        return (
            "Groq API key not configured. Set the GROQ_API_KEY "
            "environment variable."
        )

    headers = {
        "Authorization": f"Bearer {config.GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.2,   # Low temperature for factual responses
        "max_tokens": 1024,
        "top_p": 0.9,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                config.GROQ_API_URL,
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            return str(data["choices"][0]["message"]["content"])
    except httpx.HTTPStatusError as e:
        logger.error("Groq API error: %s — %s", e.response.status_code, e.response.text)
        return f"LLM API error: {e.response.status_code}"
    except Exception as e:
        logger.exception("Groq API call failed")
        return f"LLM error: {str(e)}"

    return "Unexpected error in Groq provider."


# ─── Ollama Provider ────────────────────────────────────────────

async def _call_ollama(
    system_prompt: str,
    user_message: str,
) -> str:
    """Call the local Ollama API with Llama 3.x."""
    url = f"{config.OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": config.OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 1024,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            content: str = data.get("message", {}).get("content", "No response generated.")
            return content
    except httpx.ConnectError:
        logger.error("Cannot connect to Ollama at %s", config.OLLAMA_BASE_URL)
        return (
            "Cannot connect to local Ollama instance. "
            "Make sure Ollama is running with a Llama 3 model."
        )
    except Exception as e:
        logger.exception("Ollama API call failed")
        return f"LLM error: {str(e)}"

    return "Unexpected error in Ollama provider."
