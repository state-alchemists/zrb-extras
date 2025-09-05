#!/usr/bin/env python3
"""
STT <-> TTS loop using Google GenAI (Gemini) via google-genai SDK.
Requires: set GEMINI_API_KEY or GOOGLE_API_KEY in env.

Install:
  python -m pip install --upgrade google-genai sounddevice soundfile
"""

import base64
import io
import os
import queue
import time
import tempfile
import wave
import sys

from pathlib import Path
import numpy as np
import sounddevice as sd
import soundfile as sf

from google import genai
from google.genai import types

# -------- CONFIG --------
SAMPLE_RATE = 16000
CHANNELS = 1
SILENCE_THRESHOLD = 0.01   # adjust if needed (smaller = more sensitive)
MAX_SILENCE = 4.0          # seconds of silence before stopping

# model choices (per docs)
STT_MODEL = "gemini-2.5-flash"                 # audio understanding / transcription
TTS_MODEL = "gemini-2.5-flash-preview-tts"     # TTS-capable Gemini 2.5 variant
# VOICE_NAME = "Zephyr"                          # example prebuilt voice (see docs)
VOICE_NAME = "Sulafat"
# ------------------------

# init client from env key (GEMINI_API_KEY or GOOGLE_API_KEY)
api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable.", file=sys.stderr)
    sys.exit(2)
client = genai.Client(api_key=api_key)


def record_until_silence(silence_threshold: float = SILENCE_THRESHOLD, max_silence: float = MAX_SILENCE):
    """Wait for speech to start, record, then stop after silence."""
    q = queue.Queue()
    rec_data = []

    def callback(indata, frames, time_info, status):
        q.put(indata.copy())

    print("Waiting for speech...")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=callback):
        # First detect speech
        while True:
            block = q.get()
            volume_norm = np.linalg.norm(block) / len(block)
            if volume_norm > silence_threshold:
                print("Speech detected, recording...")
                rec_data.append(block)
                break

        # Record until silence for max_silence seconds
        silence_start = None
        while True:
            block = q.get()
            rec_data.append(block)
            volume_norm = np.linalg.norm(block) / len(block)
            print(f"Volume: {volume_norm:.4f}", end='\r')

            if volume_norm < silence_threshold:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > max_silence:
                    print("\nSilence detected, stop recording.")
                    break
            else:
                silence_start = None

    # Combine into a single numpy array
    audio_data = np.concatenate(rec_data, axis=0)
    # convert to 16-bit integers
    audio_data = (audio_data * 32767).astype(np.int16)
    return audio_data


def transcribe_file(wav_path: str) -> str:
    # Upload file to Gemini Files API
    print("Uploading for transcription...")
    uploaded = client.files.upload(file=wav_path)
    # Ask model to transcribe (upload + instruction style)
    print("Requesting transcription...")
    resp = client.models.generate_content(
        model=STT_MODEL,
        contents=[uploaded, "Please transcribe the uploaded audio exactly."]
    )
    # response.text is the canonical convenience property for text outputs
    text = (resp.text or "").strip()
    print("Transcription result:", repr(text))
    return text


def synthesize_and_play(text: str):
    if not text:
        text = "I have nothing to say."
    print("Requesting TTS...")
    resp = client.models.generate_content(
        model=TTS_MODEL,
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=VOICE_NAME)
                )
            )
        )
    )
    # Extract audio blob
    try:
        part = resp.candidates[0].content.parts[0].inline_data
        b64 = part.data
        mime = getattr(part, "mime_type", None)
    except Exception as e:
        raise RuntimeError(f"No audio found in response: {e}")
    if isinstance(b64, str) and b64.startswith("data:"):
        comma = b64.find(",")
        if comma != -1:
            b64 = b64[comma + 1:]
    raw = base64.b64decode(b64) if isinstance(b64, str) else bytes(b64)
    # Prepare an in-memory WAV buffer
    buf = io.BytesIO()
    if mime and "wav" in mime.lower():
        # Already WAV or RIFF container
        buf.write(raw)
        buf.seek(0)
    else:
        # Assume raw PCM16 mono at 24kHz and wrap it
        SAMPLE_RATE_OUT = 24000
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(SAMPLE_RATE_OUT)
            wf.writeframes(raw)
        buf.seek(0)
    # Now decode and play directly from buffer
    data, sr = sf.read(buf)
    sd.play(data, sr)
    sd.wait()


def listen(
    silence_threshold: int = SILENCE_THRESHOLD, max_silence: float = MAX_SILENCE
) -> str:
    tmpdir = Path(tempfile.mkdtemp(prefix="gemini_stt_tts_"))
    in_wav = tmpdir / "input.wav"
    # normalize audio
    audio_data = record_until_silence(silence_threshold, max_silence)
    audio_data = audio_data / np.max(np.abs(audio_data))
    sf.write(str(in_wav), audio_data, SAMPLE_RATE, subtype="PCM_16")
    return transcribe_file(in_wav)
    


def speak(text: str) -> bool:
    synthesize_and_play(text)

