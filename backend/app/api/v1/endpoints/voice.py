"""
Voice processing endpoints for JARVIS.

POST /voice/transcribe   – upload audio file → transcribed text (Whisper)
POST /voice/synthesize   – text → audio file (ElevenLabs)
POST /voice/command      – full voice command pipeline (STT → AI → TTS)
WS   /voice/stream       – real-time voice streaming
GET  /voice/status       – voice pipeline component status
"""

from __future__ import annotations

import io
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, Form, UploadFile, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from backend.app.core.config import get_settings
from backend.app.core.exceptions import VoiceException
from backend.app.core.logging_config import get_logger
from backend.app.utils.helpers import generate_id, now_utc

logger = get_logger(__name__)
settings = get_settings()
router = APIRouter()

# Supported audio MIME types
_AUDIO_MIME_TYPES = {
    "audio/mpeg", "audio/mp3", "audio/mp4", "audio/wav",
    "audio/x-wav", "audio/webm", "audio/ogg", "audio/flac",
    "audio/x-m4a", "application/octet-stream",
}


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class TranscribeResponse(BaseModel):
    transcript: str
    language: Optional[str] = None
    duration_seconds: Optional[float] = None
    model_used: str
    processed_at: str


class SynthesisRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    voice_id: Optional[str] = Field(default=None, description="Override default voice")
    stability: float = Field(default=0.5, ge=0.0, le=1.0)
    similarity_boost: float = Field(default=0.75, ge=0.0, le=1.0)


class VoiceCommandRequest(BaseModel):
    # Accept text OR we process the audio bytes from form upload
    text: Optional[str] = Field(default=None, description="Pre-transcribed text (skip STT)")
    conversation_id: Optional[str] = None
    synthesize_response: bool = Field(
        default=True, description="Return TTS audio for the AI response"
    )


class VoiceCommandResponse(BaseModel):
    transcript: Optional[str]
    ai_response: str
    conversation_id: str
    audio_url: Optional[str] = None  # URL to the synthesized audio
    processing_time_ms: float


class VoiceStatusResponse(BaseModel):
    whisper: Dict[str, Any]
    elevenlabs: Dict[str, Any]
    overall: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _transcribe_audio(audio_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Transcribe *audio_bytes* using Whisper.

    Returns a dict with ``transcript``, ``language``, and ``duration``.
    Falls back to a mock result when whisper is not installed.
    """
    try:
        import whisper  # type: ignore
        import tempfile, os

        model = whisper.load_model(settings.WHISPER_MODEL)

        # Whisper requires a file path; write to a temp file.
        suffix = "." + (filename.rsplit(".", 1)[-1] if "." in filename else "wav")
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            result = model.transcribe(tmp_path)
        finally:
            os.unlink(tmp_path)

        return {
            "transcript": result.get("text", "").strip(),
            "language": result.get("language"),
            "duration": result.get("segments", [{}])[-1].get("end") if result.get("segments") else None,
        }

    except ImportError:
        logger.warning("whisper_not_installed_using_mock")
        return {
            "transcript": f"[Mock transcript of {len(audio_bytes)} bytes from {filename}]",
            "language": "en",
            "duration": len(audio_bytes) / 16000.0,  # rough estimate
        }
    except Exception as exc:
        raise VoiceException(
            message="Transcription failed.",
            code="WHISPER_ERROR",
            details={"error": str(exc)},
        ) from exc


async def _synthesize_speech(
    text: str,
    voice_id: Optional[str] = None,
    stability: float = 0.5,
    similarity_boost: float = 0.75,
) -> bytes:
    """Synthesize *text* to speech using ElevenLabs.

    Returns raw MP3 bytes.  Falls back to a silent WAV stub when the SDK
    is not installed or the API key is missing.
    """
    effective_voice = voice_id or settings.ELEVENLABS_VOICE_ID

    if not settings.ELEVENLABS_API_KEY:
        logger.warning("elevenlabs_key_missing_returning_stub_audio")
        # Return minimal valid WAV (44-byte header, no audio data)
        return _minimal_wav()

    try:
        from elevenlabs import ElevenLabs, VoiceSettings  # type: ignore

        client = ElevenLabs(api_key=settings.ELEVENLABS_API_KEY)
        audio_iterator = client.text_to_speech.convert(
            voice_id=effective_voice,
            text=text,
            model_id="eleven_monolingual_v1",
            voice_settings=VoiceSettings(
                stability=stability,
                similarity_boost=similarity_boost,
            ),
        )
        return b"".join(audio_iterator)

    except ImportError:
        logger.warning("elevenlabs_not_installed_returning_stub_audio")
        return _minimal_wav()
    except Exception as exc:
        raise VoiceException(
            message="Speech synthesis failed.",
            code="ELEVENLABS_ERROR",
            details={"error": str(exc)},
        ) from exc


def _minimal_wav() -> bytes:
    """Return a minimal valid WAV file (silence) as a stub."""
    import struct

    num_samples = 0
    channels = 1
    sample_rate = 16000
    bits_per_sample = 16
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = num_samples * block_align
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,  # subchunk1 size
        1,   # PCM
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/transcribe",
    response_model=TranscribeResponse,
    status_code=status.HTTP_200_OK,
    summary="Transcribe uploaded audio to text",
)
async def transcribe(
    audio: UploadFile = File(..., description="Audio file (mp3, wav, ogg, flac, webm, m4a)"),
) -> TranscribeResponse:
    """Accept an audio file and return a transcription produced by Whisper."""
    content_type = audio.content_type or "application/octet-stream"
    if content_type not in _AUDIO_MIME_TYPES:
        raise VoiceException(
            message=f"Unsupported audio format: {content_type}",
            code="UNSUPPORTED_FORMAT",
            details={"content_type": content_type, "supported": list(_AUDIO_MIME_TYPES)},
        )

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise VoiceException(
            message="Uploaded audio file is empty.",
            code="EMPTY_FILE",
        )

    result = await _transcribe_audio(audio_bytes, audio.filename or "audio.wav")

    return TranscribeResponse(
        transcript=result["transcript"],
        language=result.get("language"),
        duration_seconds=result.get("duration"),
        model_used=settings.WHISPER_MODEL,
        processed_at=now_utc().isoformat(),
    )


@router.post(
    "/synthesize",
    summary="Synthesize text to speech",
    response_description="Audio file (MP3 or WAV)",
    status_code=status.HTTP_200_OK,
)
async def synthesize(payload: SynthesisRequest) -> StreamingResponse:
    """Convert text to speech and stream the resulting audio bytes."""
    audio_bytes = await _synthesize_speech(
        text=payload.text,
        voice_id=payload.voice_id,
        stability=payload.stability,
        similarity_boost=payload.similarity_boost,
    )
    media_type = "audio/wav" if audio_bytes[:4] == b"RIFF" else "audio/mpeg"
    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type=media_type,
        headers={
            "Content-Disposition": "attachment; filename=speech.mp3",
            "Content-Length": str(len(audio_bytes)),
        },
    )


@router.post(
    "/command",
    response_model=VoiceCommandResponse,
    status_code=status.HTTP_200_OK,
    summary="End-to-end voice command pipeline",
)
async def voice_command(
    audio: Optional[UploadFile] = File(default=None),
    text: Optional[str] = Form(default=None),
    conversation_id: Optional[str] = Form(default=None),
    synthesize_response: bool = Form(default=True),
) -> VoiceCommandResponse:
    """Full voice pipeline: STT → AI reasoning → optional TTS.

    Either *audio* or *text* must be provided.
    """
    start_time = time.perf_counter()
    conv_id = conversation_id or generate_id()
    transcript: Optional[str] = None

    # 1. STT
    if audio is not None:
        audio_bytes = await audio.read()
        result = await _transcribe_audio(audio_bytes, audio.filename or "audio.wav")
        transcript = result["transcript"]
    elif text:
        transcript = text
    else:
        raise VoiceException(
            message="Provide either an audio file or a text input.",
            code="NO_INPUT",
        )

    # 2. AI
    ai_response_text = ""
    try:
        from openai import AsyncOpenAI  # type: ignore

        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        resp = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are JARVIS, a helpful AI assistant. Reply concisely."},
                {"role": "user", "content": transcript},
            ],
            max_tokens=512,
            stream=False,
        )
        ai_response_text = resp.choices[0].message.content or ""
    except ImportError:
        ai_response_text = f"[Mock AI response to: {transcript[:80]}]"
    except Exception as exc:
        raise VoiceException(
            message="AI processing of voice command failed.",
            code="AI_VOICE_ERROR",
            details={"error": str(exc)},
        ) from exc

    # 3. TTS (optional)
    audio_url: Optional[str] = None
    if synthesize_response:
        # In a full implementation this would store the bytes and return a URL.
        # Here we indicate it would be available.
        audio_url = f"/api/v1/voice/synthesize"  # client can POST text to get audio

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return VoiceCommandResponse(
        transcript=transcript,
        ai_response=ai_response_text.strip(),
        conversation_id=conv_id,
        audio_url=audio_url,
        processing_time_ms=round(elapsed_ms, 2),
    )


@router.websocket("/stream")
async def voice_stream(websocket: WebSocket) -> None:
    """Real-time voice streaming WebSocket.

    Protocol:
      - Client sends binary audio chunks
      - Server sends JSON: ``{"type": "partial", "transcript": "..."}``
      - Client sends ``{"type": "end"}`` JSON to signal end of utterance
      - Server sends ``{"type": "final", "transcript": "...", "response": "..."}``
    """
    await websocket.accept()
    logger.info("voice_stream_connected", client=websocket.client)
    audio_buffer = bytearray()

    try:
        while True:
            msg = await websocket.receive()

            if "bytes" in msg and msg["bytes"]:
                audio_buffer.extend(msg["bytes"])
                # Send a partial acknowledgment
                await websocket.send_json(
                    {"type": "partial", "bytes_received": len(audio_buffer)}
                )

            elif "text" in msg and msg["text"]:
                import json as _json

                try:
                    data = _json.loads(msg["text"])
                except ValueError:
                    continue

                if data.get("type") == "end":
                    if audio_buffer:
                        result = await _transcribe_audio(
                            bytes(audio_buffer), "stream.wav"
                        )
                        transcript = result["transcript"]
                        audio_buffer.clear()
                        await websocket.send_json(
                            {
                                "type": "final",
                                "transcript": transcript,
                                "response": f"Received: {transcript}",
                            }
                        )
                    else:
                        await websocket.send_json(
                            {"type": "error", "message": "No audio received."}
                        )

    except WebSocketDisconnect:
        logger.info("voice_stream_disconnected")
    except Exception as exc:
        logger.error("voice_stream_error", error=str(exc))
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass


@router.get(
    "/status",
    response_model=VoiceStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Voice pipeline component status",
)
async def voice_status() -> VoiceStatusResponse:
    """Report availability and configuration of Whisper and ElevenLabs."""
    # Whisper
    whisper_info: Dict[str, Any] = {"model": settings.WHISPER_MODEL}
    try:
        import whisper  # type: ignore  # noqa: F401

        whisper_info["status"] = "available"
    except ImportError:
        whisper_info["status"] = "not_installed"

    # ElevenLabs
    elevenlabs_info: Dict[str, Any] = {"voice_id": settings.ELEVENLABS_VOICE_ID}
    if not settings.ELEVENLABS_API_KEY:
        elevenlabs_info["status"] = "not_configured"
    else:
        try:
            from elevenlabs import ElevenLabs  # type: ignore  # noqa: F401

            elevenlabs_info["status"] = "available"
        except ImportError:
            elevenlabs_info["status"] = "not_installed"

    overall = (
        "ok"
        if whisper_info["status"] == "available"
        and elevenlabs_info["status"] == "available"
        else "degraded"
    )

    return VoiceStatusResponse(
        whisper=whisper_info,
        elevenlabs=elevenlabs_info,
        overall=overall,
    )
