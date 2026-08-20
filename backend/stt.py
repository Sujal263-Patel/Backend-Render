"""Sarvam Saarika v2 speech-to-text adapter used by the voice pipeline."""
from __future__ import annotations

import io
import requests

from config import settings


class STTError(Exception):
    pass


def transcribe(audio_bytes: bytes, filename: str = "audio.wav") -> dict:
    """Returns {"text": str, "provider": str, "raw": dict}"""
    return _transcribe_sarvam(audio_bytes, filename)


def _transcribe_sarvam(audio_bytes: bytes, filename: str) -> dict:
    if not settings.SARVAM_API_KEY:
        raise STTError(
            "SARVAM_API_KEY not set. Export it in the environment before "
            "starting the server (see README)."
        )
    files = {"file": (filename, io.BytesIO(audio_bytes), "audio/wav")}
    data = {"language_code": settings.SARVAM_LANGUAGE_CODE, "model": "saarika:v2"}
    headers = {"api-subscription-key": settings.SARVAM_API_KEY}
    try:
        resp = requests.post(
            settings.SARVAM_STT_URL, headers=headers, files=files, data=data, timeout=15
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        raise STTError(f"Sarvam STT request failed: {e}") from e

    payload = resp.json()
    text = payload.get("transcript", "")
    if not text:
        raise STTError("Sarvam STT returned an empty transcript")
    return {"text": text, "provider": "sarvam", "raw": payload}

