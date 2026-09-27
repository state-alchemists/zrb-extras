from __future__ import annotations

import shutil

from zrb.llm.speech import AnySpeechBackend, Utterance

from zrb_extras.llm.tool.termux.speak import create_termux_tts_command


class TermuxSpeechBackend(AnySpeechBackend):
    """Android's text-to-speech through Termux:API's `termux-tts-speak`.

    Each argument maps to one of its flags: *language* ``-l``, *voice_name*
    ``-v`` (variant), *engine* ``-e``, *region* ``-n``, *rate* ``-r``,
    *pitch* ``-p``, *stream* ``-s``.
    """

    def __init__(
        self,
        language: str | None = None,
        voice_name: str | None = None,
        engine: str | None = None,
        region: str | None = None,
        rate: float | None = None,
        pitch: float | None = None,
        stream: str | None = None,
    ) -> None:
        self._options = dict(
            language=language,
            voice_name=voice_name,
            engine=engine,
            region=region,
            rate=rate,
            pitch=pitch,
            stream=stream,
        )

    @property
    def name(self) -> str:
        return "termux"

    def create_utterance(self, text: str) -> Utterance:
        if shutil.which("termux-tts-speak") is None:
            raise RuntimeError(
                "termux-tts-speak not on PATH: pkg install termux-api, and "
                "install the Termux:API app"
            )
        return Utterance(create_termux_tts_command(text, **self._options))
