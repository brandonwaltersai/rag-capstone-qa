"""OpenAI client with retry/quota handling. Reads OPENAI_API_KEY from the environment (.env)."""
import os
import time
import random
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def ensure_openai_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY missing. Copy .env.example to .env and add your key."
        )
    return key


def is_quota_error(e: Exception) -> bool:
    msg = str(e).lower()
    return ("insufficient_quota" in msg) or ("exceeded your current quota" in msg) or ("check your plan and billing" in msg)


def is_transient_error(e: Exception) -> bool:
    if is_quota_error(e):
        return False
    msg = str(e).lower()
    status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
    if status in {429, 500, 502, 503, 504}:
        return True
    return any(x in msg for x in ["timeout", "temporarily", "overloaded", "try again", "connection"])


def call_with_retries(fn, operation: str, max_retries: int = 5, base_delay: float = 1.0):
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as e:
            if is_quota_error(e):
                raise RuntimeError(
                    f"{operation} failed: OpenAI quota/billing issue. "
                    "Enable pay-as-you-go billing or reduce KB_SAMPLE_N for a smaller pilot run."
                ) from e
            if attempt >= max_retries or not is_transient_error(e):
                raise
            sleep_s = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.25)
            time.sleep(sleep_s)


_ = ensure_openai_api_key()
client = OpenAI()
