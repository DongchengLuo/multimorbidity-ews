# -*- coding: utf-8 -*-
"""Unified LLM configuration entry point.

Background: modules read os.getenv("LOCAL_LLM_MODEL_NAME") at import, but only
batch_processor.py calls load_dotenv. If any module is imported before batch_processor
(e.g. a test script imports report_generator first), env vars are not yet injected and it silently falls back to
a hard-coded default model name, causing "the .env says dsv41-flash-1017 but a different model is actually called".

This module unconditionally loads the project-root .env at import and provides a single default model name,
All modules that call the LLM should read configuration from here.
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
except Exception:  # pragma: no cover - fall back to plain env vars when dotenv is missing
    pass

# When migrating to a new server, set LOCAL_LLM_API_URL / LOCAL_LLM_MODEL_NAME in .env.
# No intranet address is built in, avoiding silent connections to unreachable legacy endpoints.
DEFAULT_MODEL = ""
DEFAULT_API_URL = ""

LLM_API_URL = (os.getenv("LOCAL_LLM_API_URL") or DEFAULT_API_URL).strip()
LLM_BASE_URL = LLM_API_URL.rsplit("/chat/completions", 1)[0] if LLM_API_URL else ""
LLM_MODEL = (os.getenv("LOCAL_LLM_MODEL_NAME") or DEFAULT_MODEL).strip()
LLM_API_KEY = os.getenv("LOCAL_LLM_API_KEY") or "EMPTY"


def require_llm_config():
    """Validate configuration before calling the LLM; raise an actionable error when missing."""
    missing = []
    if not LLM_API_URL:
        missing.append("LOCAL_LLM_API_URL")
    if not LLM_MODEL:
        missing.append("LOCAL_LLM_MODEL_NAME")
    if missing:
        raise RuntimeError(
            "LLM not configured: set "
            + " / ".join(missing)
            + " in the project-root .env (see .env.example). Current value: "
            + f"LOCAL_LLM_API_URL={LLM_API_URL!r}, LOCAL_LLM_MODEL_NAME={LLM_MODEL!r}"
        )
    return LLM_API_URL, LLM_MODEL, LLM_API_KEY
