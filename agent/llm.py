from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

# Ensure environment variables are loaded even when agent.llm is imported standalone
_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_ROOT / ".env", override=False)

DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL","").strip() #"llama-3.3-70b-versatile"


def model_name() -> str:
    val = os.getenv("GROQ_MODEL", "").strip()
    if not val or val == "openai/gpt-oss-120b":
        return DEFAULT_GROQ_MODEL
    return val


@lru_cache(maxsize=1)
def get_groq_client() -> Any:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key or api_key.lower().startswith("replace_") or api_key.startswith("gsk_Not"):
        raise RuntimeError(
            "GROQ_API_KEY is missing or invalid. Configure a valid Groq API key in .env (get one from https://console.groq.com/)."
        )
    try:
        from groq import Groq
    except ImportError as exc:
        raise RuntimeError("Groq SDK is not installed. Run: pip install -r requirements.txt") from exc
    return Groq(api_key=api_key, timeout=float(os.getenv("GROQ_TIMEOUT", "120")))


def groq_chat(messages: list[dict[str, Any]], **kwargs: Any):
    """Call the configured Groq model."""
    model = kwargs.pop("model", None) or model_name()
    return get_groq_client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        **kwargs,
    )


def conversation_response(messages: list[dict[str, str]]) -> str:
    context = [{"role": "system", "content": (
        "You are the conversational assistant inside an image-editing application. "
        "Be concise and helpful. Do not claim an image was edited unless the tool workflow confirms it. "
        "If the user asks to edit an image but no image is available, ask them to upload one."
    )}]
    context.extend(messages[-12:])
    response = groq_chat(context, max_tokens=900)
    content = response.choices[0].message.content
    return (content or "I couldn't generate a response. Please try again.").strip()

