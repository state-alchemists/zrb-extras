SAMPLE_RATE = 16000
CHANNELS = 1
SILENCE_THRESHOLD = 0.01  # adjust if needed (smaller = more sensitive)
MAX_SILENCE = 4.0  # seconds of silence before stopping

# model choices (per docs)
STT_MODEL = "gemini-2.5-flash"  # audio understanding / transcription
TTS_MODEL = "gemini-2.5-flash-preview-tts"  # TTS-capable Gemini 2.5 variant
VOICE_NAME = "Sulafat"
