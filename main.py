from __future__ import annotations

import logging
import os
import tempfile
from importlib import import_module
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("opic-practice-stt")

app = FastAPI(title="opic-practice-stt")

DEFAULT_MODEL_SIZE = os.getenv("STT_MODEL_SIZE", "base.en")
DEFAULT_DEVICE = os.getenv("STT_DEVICE", "cpu")
DEFAULT_COMPUTE_TYPE = os.getenv("STT_COMPUTE_TYPE", "int8")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/stt/transcriptions")
async def transcribe_audio(
    audioFile: UploadFile = File(...),
    language: str = Form("en"),
    questionId: str | None = Form(None),
) -> dict[str, Any]:
    if not audioFile.content_type or not audioFile.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="audioFile must be an audio file.")

    audio_bytes = await audioFile.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="audioFile must not be empty.")

    transcript = transcribe_audio_bytes(
        audio_bytes=audio_bytes,
        content_type=audioFile.content_type,
        language=language,
    )

    logger.info(
        "Processed STT upload questionId=%s fileName=%s contentType=%s size=%s transcriptLength=%s",
        questionId,
        audioFile.filename,
        audioFile.content_type,
        len(audio_bytes),
        len(transcript),
    )

    return {
        "transcript": transcript,
        "language": language,
        "questionId": questionId,
        "provider": "faster-whisper",
        "fileName": audioFile.filename,
        "contentType": audioFile.content_type,
        "fileSize": len(audio_bytes),
    }


@lru_cache(maxsize=1)
def get_whisper_model() -> Any:
    try:
        whisper_module = import_module("faster_whisper")
        whisper_model_cls = whisper_module.WhisperModel
    except Exception as exc:
        logger.exception("Failed to import faster-whisper runtime")
        raise HTTPException(
            status_code=503,
            detail=(
                "STT service is unavailable in this environment. "
                f"faster-whisper runtime could not be loaded: {exc}"
            ),
        ) from exc

    logger.info(
        "Loading faster-whisper model model_size=%s device=%s compute_type=%s",
        DEFAULT_MODEL_SIZE,
        DEFAULT_DEVICE,
        DEFAULT_COMPUTE_TYPE,
    )
    return whisper_model_cls(
        DEFAULT_MODEL_SIZE,
        device=DEFAULT_DEVICE,
        compute_type=DEFAULT_COMPUTE_TYPE,
    )


def transcribe_audio_bytes(audio_bytes: bytes, content_type: str | None, language: str) -> str:
    suffix = guess_file_suffix(content_type)
    normalized_language = normalize_language(language)

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
        temp_audio.write(audio_bytes)
        temp_audio_path = temp_audio.name

    try:
        model = get_whisper_model()
        segments, info = model.transcribe(
            temp_audio_path,
            beam_size=5,
            language=normalized_language,
            vad_filter=True,
        )
        segment_list = list(segments)
        transcript = " ".join(
            segment.text.strip()
            for segment in segment_list
            if segment.text and segment.text.strip()
        ).strip()

        logger.info(
            "faster-whisper completed language=%s probability=%.4f segments=%s",
            info.language,
            info.language_probability,
            len(segment_list),
        )

        if not transcript:
            raise HTTPException(status_code=422, detail="Speech was detected but no transcript was produced.")

        return transcript
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("faster-whisper transcription failed")
        raise HTTPException(status_code=500, detail=f"STT transcription failed: {exc}") from exc
    finally:
        try:
            os.remove(temp_audio_path)
        except OSError:
            logger.warning("Failed to remove temporary audio file: %s", temp_audio_path)


def normalize_language(language: str | None) -> str:
    if not language:
        return "en"

    normalized = language.strip().lower()
    if normalized in {"english", "en-us", "en-gb"}:
        return "en"
    return normalized or "en"


def guess_file_suffix(content_type: str | None) -> str:
    if not content_type:
        return ".webm"

    if "ogg" in content_type:
        return ".ogg"
    if "mp4" in content_type or "m4a" in content_type:
        return ".mp4"
    if "wav" in content_type:
        return ".wav"
    if "mpeg" in content_type or "mp3" in content_type:
        return ".mp3"
    return ".webm"
