import base64
import io
import os
import queue
import sys
import tempfile
import time
import wave
from collections import deque
from pathlib import Path
from typing import Callable

import numpy as np
import sounddevice as sd
import soundfile as sf
from google import genai
from google.genai import types

SAMPLE_RATE = 16000
CHANNELS = 1
SILENCE_THRESHOLD = 0.01  # adjust if needed (smaller = more sensitive)
MAX_SILENCE = 4.0  # seconds of silence before stopping

# model choices (per docs)
STT_MODEL = "gemini-2.5-flash"  # audio understanding / transcription
TTS_MODEL = "gemini-2.5-flash-preview-tts"  # TTS-capable Gemini 2.5 variant
# VOICE_NAME = "Zephyr"                          # example prebuilt voice (see docs)
VOICE_NAME = "Sulafat"


def create_speak_tool(
    client: genai.Client | None = None,
    api_key: str | None = None,
    tts_model: str = TTS_MODEL,
    voice_name: str = VOICE_NAME,
    sample_rate_out: int = 24000,
) -> Callable[[str], bool]:
    def speak(text: str) -> bool:
        """
        Converts a given text or SSML document into speech and plays it out loud.

        Use this tool to communicate with the user verbally.
        Keep the text concise (a sentence or two) to ensure a fast response.

        You can control the speech's intonation, rate, and emphasis by providing
        the text as an SSML (Speech Synthesis Markup Language) document.
        To do this, wrap the text in <speak> tags.

        SSML Example:
        '<speak>
          I can speak <emphasis>very</emphasis> slowly
          <prosody rate="slow">or very quickly.</prosody>
        </speak>'

        Args:
            text: The plain text or SSML string to be converted to speech.

        Returns:
            True if the speech was successfully generated and played, False otherwise.
        """
        return _synthesize_and_play(
            client=_get_client(client, api_key),
            text=text,
            tts_model=tts_model,
            voice_name=voice_name,
            sample_rate_out=sample_rate_out,
        )

    return speak


def _synthesize_and_play(
    client: genai.Client,
    text: str,
    tts_model: str,
    voice_name: str,
    sample_rate_out: int,
):
    if not text:
        text = "I have nothing to say."
    print("Requesting TTS...", file=sys.stderr)
    resp = client.models.generate_content(
        model=tts_model,
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice_name
                    )
                )
            ),
        ),
    )
    # Extract audio blob
    print("Extracting audio...", file=sys.stderr)
    try:
        part = resp.candidates[0].content.parts[0].inline_data
        b64 = part.data
        mime = getattr(part, "mime_type", None)
    except Exception as e:
        raise RuntimeError(f"No audio found in response: {e}")
    if isinstance(b64, str) and b64.startswith("data:"):
        comma = b64.find(",")
        if comma != -1:
            b64 = b64[comma + 1 :]
    raw = base64.b64decode(b64) if isinstance(b64, str) else bytes(b64)
    # Prepare an in-memory WAV buffer
    print("Preparing audio buffer...", file=sys.stderr)
    buf = io.BytesIO()
    if mime and "wav" in mime.lower():
        # Already WAV or RIFF container
        buf.write(raw)
        buf.seek(0)
    else:
        # Assume raw PCM16 mono at 24kHz and wrap it
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate_out)
            wf.writeframes(raw)
        buf.seek(0)
    # Now decode and play directly from buffer
    data, sr = sf.read(buf)
    print("Playing audio...", file=sys.stderr)
    sd.play(data, sr)
    sd.wait()
    return True


def create_listen_tool(
    client: genai.Client | None = None,
    api_key: str | None = None,
    stt_model: str = STT_MODEL,
    sample_rate: int = SAMPLE_RATE,
    channels: int = CHANNELS,
    silence_threshold: float = SILENCE_THRESHOLD,
    max_silence: float = MAX_SILENCE,
) -> Callable[[], str]:
    """
    Factory to create a configurable listen tool.
    """

    def listen() -> str:
        """
        Records audio from the microphone, waits for silence,
        and then transcribes the speech to text.

        This tool is used to capture the user's verbal response.
        It automatically handles silence detection to determine when the
        user has finished speaking.

        Returns:
            The transcribed text from the user's speech.
        """
        # Warm up the sound device to prevent ALSA timeout
        with sd.Stream(samplerate=sample_rate, channels=channels):
            pass

        tmpdir = Path(tempfile.mkdtemp(prefix="gemini_stt_tts_"))
        in_wav = tmpdir / "input.wav"
        # Record audio
        audio_data = _record_until_silence(
            sample_rate=sample_rate,
            channels=channels,
            silence_threshold=silence_threshold,
            max_silence=max_silence,
        )
        # Normalize and write to file
        audio_data = audio_data / np.max(np.abs(audio_data))
        sf.write(str(in_wav), audio_data, sample_rate, subtype="PCM_16")
        # Transcribe
        return _transcribe_file(
            client=_get_client(client, api_key),
            wav_path=str(in_wav),
            stt_model=stt_model,
        )

    return listen


def _record_until_silence(
    sample_rate: int,
    channels: int,
    silence_threshold: float,
    max_silence: float,
):
    """Wait for speech to start, record, then stop after silence."""
    q = queue.Queue()
    rec_data = []
    PRE_BUFFER_DURATION = 0.5  # seconds
    pre_buffer_size = int(PRE_BUFFER_DURATION * sample_rate / 1024)  # in blocks
    pre_buffer = deque(maxlen=pre_buffer_size)

    def callback(indata, frames, time_info, status):
        q.put(indata.copy())

    print("Waiting for speech...", file=sys.stderr)
    with sd.InputStream(samplerate=sample_rate, channels=channels, callback=callback):
        # First detect speech
        while True:
            block = q.get()
            pre_buffer.append(block)
            volume_norm = np.linalg.norm(block) / len(block)
            if volume_norm > silence_threshold:
                print("Speech detected, recording...", file=sys.stderr)
                rec_data.extend(pre_buffer)
                rec_data.append(block)
                break

        # Record until silence for max_silence seconds
        silence_start = None
        while True:
            block = q.get()
            rec_data.append(block)
            volume_norm = np.linalg.norm(block) / len(block)
            print("Volume: {volume_norm:.4f}", end="\r", file=sys.stderr)
            if volume_norm < silence_threshold:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > max_silence:
                    print("\nSilence detected, stop recording.", file=sys.stderr)
                    break
            else:
                silence_start = None

    # Combine into a single numpy array
    audio_data = np.concatenate(rec_data, axis=0)
    # convert to 16-bit integers
    audio_data = (audio_data * 32767).astype(np.int16)
    return audio_data


def _transcribe_file(
    client: genai.Client,
    wav_path: str,
    stt_model: str,
) -> str:
    # Upload file to Gemini Files API
    print("Uploading for transcription...", file=sys.stderr)
    uploaded = client.files.upload(file=wav_path)
    # Ask model to transcribe (upload + instruction style)
    print("Requesting transcription...", file=sys.stderr)
    resp = client.models.generate_content(
        model=stt_model,
        contents=[uploaded, "Please transcribe the uploaded audio exactly."],
    )
    # response.text is the canonical convenience property for text outputs
    text = (resp.text or "").strip()
    print("Transcription result:", repr(text), file=sys.stderr)
    return text


def _get_client(
    client: genai.Client | None = None,
    api_key: str | None = None,
) -> genai.Client:
    if client is not None:
        return client
    if api_key is None:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    return genai.Client(api_key=api_key)
