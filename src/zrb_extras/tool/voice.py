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
import re
import time
import tempfile
import wave
import sys
from concurrent.futures import ThreadPoolExecutor

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
STT_MODEL = "gemini-1.5-flash"                 # audio understanding / transcription
TTS_MODEL = "models/text-to-speech"     # TTS-capable Gemini 1.5 variant
VOICE_NAME = "Echo"                          # example prebuilt voice (see docs)
# ------------------------

# init client from env key (GEMINI_API_KEY or GOOGLE_API_KEY)
api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable.", file=sys.stderr)
    sys.exit(2)
genai.configure(api_key=api_key)


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
    model = genai.GenerativeModel(model_name=STT_MODEL)
    uploaded_file = genai.upload_file(path=wav_path)
    # Ask model to transcribe (upload + instruction style)
    print("Requesting transcription...")
    resp = model.generate_content(
        [uploaded_file, "Please transcribe the uploaded audio exactly."]
    )
    # response.text is the canonical convenience property for text outputs
    text = (resp.text or "").strip()
    print("Transcription result:", repr(text))
    return text


def _synthesize_chunk(text: str):
    if not text:
        return None
    print(f"Requesting TTS for: '{text}'")
    try:
        resp = genai.text_to_speech(
            model=TTS_MODEL,
            text=text,
            voice_name=VOICE_NAME,
        )
        return resp.audio_data
    except Exception as e:
        print(f"TTS failed for '{text}': {e}", file=sys.stderr)
        return None


def split_into_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    # Super simple sentence splitter
    sentences = re.split(r'(?<=[.?!])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def synthesize_and_play(text: str):
    if not text:
        text = "I have nothing to say."

    sentences = split_into_sentences(text)
    if not sentences:
        if text:
            sentences = [text]
        else:
            return

    audio_queue = queue.Queue()
    stop_playback = False

    def producer(executor: ThreadPoolExecutor):
        # Submit all sentences for synthesis
        futures = [executor.submit(_synthesize_chunk, s) for s in sentences]
        for future in futures:
            audio_chunk = future.result()
            if audio_chunk:
                audio_queue.put(audio_chunk)
        audio_queue.put(None)  # Sentinel to stop consumer

    def consumer():
        nonlocal stop_playback
        while not stop_playback:
            try:
                audio_chunk = audio_queue.get(timeout=0.1)
                if audio_chunk is None:
                    break
                buf = io.BytesIO(audio_chunk)
                data, sr = sf.read(buf)
                sd.play(data, sr)
                sd.wait()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error during playback: {e}", file=sys.stderr)
                break

    with ThreadPoolExecutor(max_workers=4) as executor:
        prod_thread = executor.submit(producer, executor)
        cons_thread = executor.submit(consumer)

        try:
            # Wait for producer to finish
            prod_thread.result()
            # Wait for consumer to finish
            cons_thread.result()
        except KeyboardInterrupt:
            print("\nInterrupted by user.", file=sys.stderr)
            stop_playback = True
            # Drain the queue
            while not audio_queue.empty():
                try:
                    audio_queue.get_nowait()
                except queue.Empty:
                    break
            # Wait for threads to finish up
            prod_thread.result()
            cons_thread.result()


def listen(
    silence_threshold: int = SILENCE_THRESHOLD, max_silence: float = MAX_SILENCE
) -> str:
    tmpdir = Path(tempfile.mkdtemp(prefix="gemini_stt_tts_"))
    in_wav = tmpdir / "input.wav"
    # normalize audio
    audio_data = record_until_silence(silence_threshold, max_silence)
    if np.max(np.abs(audio_data)) > 0:
        audio_data = audio_data / np.max(np.abs(audio_data))
    sf.write(str(in_wav), audio_data, SAMPLE_RATE, subtype="PCM_16")
    return transcribe_file(str(in_wav))


def speak(text: str):
    synthesize_and_play(text)


