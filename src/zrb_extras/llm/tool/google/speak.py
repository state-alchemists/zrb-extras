import asyncio
import base64
import io
import wave
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from typing_extensions import TypedDict

from zrb_extras.llm.tool.google.client import get_client
from zrb_extras.llm.tool.google.default_config import TTS_MODEL, VOICE_NAME

if TYPE_CHECKING:
    from google import genai
    from google.genai import types


class MultiSpeakerVoice(TypedDict):
    speaker: str
    voice: str


def create_speak_tool(
    client: "genai.Client | None" = None,
    api_key: str | None = None,
    tts_model: str | None = None,
    voice_name: str | list[MultiSpeakerVoice] | None = None,
    sample_rate_out: int | None = None,
    safety_settings: "list[types.SafetySetting] | None" = None,
    tool_name: str | None = None,
    tool_description: str | None = None,
) -> Callable[
    [str, str | list[MultiSpeakerVoice] | None], Coroutine[Any, Any, bool]
]:
    tts_model = tts_model or TTS_MODEL
    voice_name = voice_name or VOICE_NAME
    sample_rate_out = sample_rate_out if sample_rate_out is not None else 24000

    async def speak(
        text: str,
        voice_name: str | list[MultiSpeakerVoice] | None = voice_name,
    ) -> bool:
        """Converts text to speech and plays it aloud.

        Use this tool to verbally communicate with the user.
        The system will play the generated audio and return `True` upon completion.
        Keep the text concise (a sentence or two) for a faster response.

        You can control the speech's style by describing how you want the text to be spoken.
        For example: 'Say in a cheerful, enthusiastic voice: "Good morning, everyone!"'

        Args:
          text: The text to be spoken. Can be plain text or a controllable prompt.
          voice_name: The voice or voices to use.
            - For a single speaker, provide a voice name string (e.g., "Sulafat").
            - For multiple speakers (up to two), provide a list of speaker-voice mappings.
              Example: `[{"speaker": "User", "voice": "Aoede"}, {"speaker": "Agent", "voice": "Puck"}]`

          Available voices:
          - Zephyr (Female, Bright)
          - Puck (Male, Upbeat)
          - Charon (Male, Informative)
          - Kore (Female, Firm)
          - Fenrir (Male, Excitable)
          - Leda (Female, Youthful)
          - Orus (Male, Firm)
          - Aoede (Female, Breezy)
          - Callirrhoe (Female, Easy-going)
          - Autonoe (Female, Bright)
          - Enceladus (Male, Breathy)
          - Iapetus (Male, Clear)
          - Umbriel (Male, Easy-going)
          - Algieba (Male, Smooth)
          - Despina (Female, Smooth)
          - Erinome (Female, Clear)
          - Algenib (Male, Gravelly)
          - Rasalgethi (Male, Informative)
          - Laomedeia (Female, Upbeat)
          - Achernar (Female, Soft)
          - Alnilam (Male, Firm)
          - Schedar (Male, Even)
          - Gacrux (Female, Mature)
          - Pulcherrima (Female, Forward)
          - Achird (Male, Friendly)
          - Zubenelgenubi (Male, Casual)
          - Vindemiatrix (Female, Gentle)
          - Sadachbia (Male, Lively)
          - Sadaltager (Male, Knowledgeable)
          - Sulafat (Female, Warm)
        Returns:
          True if speech was successfully generated and played, False otherwise.
        """
        return await _synthesize_and_play(
            client=get_client(client, api_key),
            text=text,
            tts_model=tts_model,
            voice_name=voice_name,
            sample_rate_out=sample_rate_out,
            safety_settings=safety_settings,
        )

    if tool_name is not None:
        speak.__name__ = tool_name
    if tool_description is not None:
        speak.__doc__ = tool_description
    return speak


async def _synthesize_and_play(
    client: "genai.Client",
    text: str,
    tts_model: str,
    voice_name: str | list[MultiSpeakerVoice] | None = None,
    sample_rate_out: int = 24000,
    safety_settings: "list[types.SafetySetting] | None" = None,
):
    try:
        import sounddevice as sd
        import soundfile as sf
        from google.genai import types
    except ImportError:
        raise ImportError(
            "google-genai dependencies are not installed. Please install zrb-extras[google-genai] or zrb-extras[all]."
        )

    if not text:
        text = "I have nothing to say."
    print("Requesting TTS...")
    resp = client.models.generate_content(
        model=tts_model,
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            safety_settings=safety_settings,
            speech_config=(
                types.SpeechConfig(
                    multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                        speaker_voice_configs=[
                            types.SpeakerVoiceConfig(
                                speaker=config.get("speaker", f"Speaker {idx + 1}"),
                                voice_config=types.VoiceConfig(
                                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                        voice_name=config.get("voice", VOICE_NAME)
                                    )
                                ),
                            )
                            for idx, config in enumerate(voice_name)
                        ]
                    )
                )
                if isinstance(voice_name, list)
                else types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice_name or VOICE_NAME
                        )
                    )
                )
            ),
        ),
    )
    # Extract audio blob
    print("Extracting audio...")
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
    print("Preparing audio buffer...")
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
    print("Playing audio...")
    sd.play(data, sr)
    await asyncio.to_thread(sd.wait)
    return True
