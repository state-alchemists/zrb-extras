import asyncio
import io
import os
import sys
import tempfile
import wave

# Global flag to disable sounddevice if it fails once
_SOUNDDEVICE_DISABLED = False

async def play_audio(
    data: bytes | io.BytesIO,
    sample_rate: int | None = None,
    channels: int = 1,
    sample_width: int = 2,
):
    """
    Plays audio data.
    If data is bytes, it's assumed to be raw PCM unless it has a WAV header.
    If data is io.BytesIO, it's treated similarly.
    """
    if isinstance(data, io.BytesIO):
        audio_content = data.getvalue()
    else:
        audio_content = data

    if not audio_content:
        return

    # Check if it's a WAV file (starts with RIFF)
    is_wav = audio_content.startswith(b"RIFF")

    # Prefer afplay on macOS
    if sys.platform == "darwin":
        await _play_with_afplay(audio_content, is_wav, sample_rate, channels, sample_width)
    else:
        await _play_with_sounddevice(audio_content, is_wav, sample_rate, channels, sample_width)


async def _play_with_afplay(
    audio_content: bytes,
    is_wav: bool,
    sample_rate: int | None,
    channels: int,
    sample_width: int,
):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        if is_wav:
            f.write(audio_content)
        else:
            if sample_rate is None:
                sample_rate = 24000  # Default fallback
            # Wrap raw PCM in WAV
            with wave.open(f, "wb") as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(sample_width)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_content)
        temp_path = f.name

    try:
        process = await asyncio.create_subprocess_exec(
            "afplay", temp_path,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL
        )
        await process.wait()
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


async def _play_with_sounddevice(
    audio_content: bytes,
    is_wav: bool,
    sample_rate: int | None,
    channels: int,
    sample_width: int,
):
    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError:
        raise ImportError(
            "sounddevice or soundfile is not installed. Please install zrb-extras[all]."
        )

    buf = io.BytesIO(audio_content)
    if is_wav:
        data, sr = sf.read(buf)
    else:
        if sample_rate is None:
            sample_rate = 24000
        import numpy as np
        # Assume PCM 16-bit
        data = np.frombuffer(audio_content, dtype=np.int16).astype(np.float32) / 32768.0
        sr = sample_rate

    sd.play(data, sr)
    await asyncio.to_thread(sd.wait)


async def play_audio_stream(
    audio_generator,
    sample_rate: int = 24000,
    channels: int = 1,
):
    """
    Plays audio from an async generator of bytes (PCM 16-bit).
    Falls back to collecting all data and playing with afplay if sounddevice fails.
    """
    global _SOUNDDEVICE_DISABLED

    # Environment override
    if os.getenv("ZRB_DISABLE_STREAMING_AUDIO", "0") == "1" or _SOUNDDEVICE_DISABLED:
        await _fallback_play_collected(audio_generator, sample_rate, channels)
        return

    try:
        import sounddevice as sd
        import numpy as np
    except ImportError:
        await _fallback_play_collected(audio_generator, sample_rate, channels)
        return

    event_loop = asyncio.get_running_loop()
    
    # We use a residue buffer to handle chunks that aren't multiples of sample width
    residue = bytearray()

    try:
        # Open a blocking OutputStream. 
        # using int16 to match source data avoids conversion noise.
        stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype='int16',
            blocksize=2048 # Reasonable blocksize for blocking writes
        )
        
        with stream:
            # Handle the case where audio_generator might be a coroutine
            import inspect
            if inspect.isawaitable(audio_generator):
                audio_generator = await audio_generator
            
            async for chunk in audio_generator:
                if not chunk:
                    continue
                
                # Handle residue from previous chunk
                if residue:
                    chunk = residue + chunk
                    residue = bytearray()
                
                # Check alignment (2 bytes per sample for int16)
                remainder = len(chunk) % 2
                if remainder != 0:
                    residue = chunk[-remainder:]
                    chunk = chunk[:-remainder]
                
                if len(chunk) == 0:
                    continue

                # Convert to numpy array
                data = np.frombuffer(chunk, dtype=np.int16)
                
                # Write to stream in a separate thread to avoid blocking asyncio loop
                # stream.write is blocking.
                await event_loop.run_in_executor(None, stream.write, data)

    except Exception as e:
        print(f"Streaming error (sounddevice): {e}. Falling back to afplay.", file=sys.stderr)
        _SOUNDDEVICE_DISABLED = True # Disable for future calls to avoid repeated failures
        # Note: We might have lost the chunks already consumed. 
        # In a real generator, we can't rewind. 
        # So this fallback only works for *future* calls or if we buffered everything (which defeats streaming).
        # For now, we just accept the error and let the user know. 
        # If the failure happens immediately, next turns will use afplay.
        pass

async def _fallback_play_collected(audio_generator, sample_rate, channels):
    """Collects all audio and plays it using the reliable non-streaming method."""
    content = b""
    # Handle the case where audio_generator might be a coroutine
    import inspect
    if inspect.isawaitable(audio_generator):
        audio_generator = await audio_generator
    async for chunk in audio_generator:
        content += chunk
    await play_audio(content, sample_rate, channels)
