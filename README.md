# Zrb extras

zrb-extras is a [pypi](https://pypi.org) package.

You can install zrb-extras by invoking the following command:

```bash
pip install zrb-extras            # then pick the extras for your backend:
pip install "zrb-extras[openai]"  # or [google-genai], [pyttsx3], [youtube], [all]
```

## Let your LLM task `speak`

`create_speak_tool` gives the agent a tool that says text aloud. The agent decides what to say and when, which suits an `LLMTask` that runs unattended ("the build finished") or a persona that should talk rather than print.

Talking *to* zrb is built into zrb itself since 3.10: `/voice` and `/handsfree` dictate, and hands-free can answer tool approvals. zrb can also read every reply aloud (`ZRB_LLM_SPEECH_ENABLED=on`). See zrb's [Voice and Camera](https://github.com/state-alchemists/zrb/blob/main/docs/llm/voice-camera.md) guide. The `listen` tool that used to live here was removed in zrb-extras 3.1.0 for that reason.

> Use the `speak` tool **or** zrb's built-in speech, not both, or replies are said twice.

### Or: Termux and pyttsx3 voices for zrb's built-in speech

zrb reads replies aloud with `say`, `espeak-ng`, OpenAI or Gemini. zrb-extras adds two more backends for it: `TermuxSpeechBackend` (Android's voices through Termux:API) and `Pyttsx3SpeechBackend` (offline, in process; extra `[pyttsx3]`). If one fails, zrb's local engine speaks instead.

```python
from zrb.builtin import llm_chat
from zrb.llm.speech import SpeechConfig, enable_speech
from zrb_extras.llm.speech import TermuxSpeechBackend

enable_speech(
    llm_chat,
    SpeechConfig(backend=TermuxSpeechBackend(language="en", rate=1.1), enabled=True),
)
```

Calling `enable_speech` again replaces the built-in call, so there is still one speaker and one `/speech` command.

### Prerequisites

#### Termux

> First of all, make sure termux has permission to access the speaker

```bash
pkg update && pkg upgrade -y
pkg install pulseaudio termux-api -y
```

Run the following script or add it to `~/.bashrc`

```bash
# start PulseAudio daemon
pulseaudio --start --load="module-native-protocol-tcp auth-ip-acl=127.0.0.1 auth-anonymous=1" --exit-idle-time=-1

# Start proot-distro
proot-distro login ubuntu
```

#### Proot-distro (Ubuntu)

```bash
apt install libasound2-dev portaudio19-dev pulseaudio
```

### Create `zrb_init.py`

```python
import os

from zrb.builtin import llm_chat
from zrb.llm.tool_call.tool_policy.auto_approve import auto_approve
from zrb_extras.llm.tool import create_speak_tool

# Valid modes: "google", "openai", "termux", "pyttsx3"
VOICE_MODE = os.getenv("VOICE_MODE", "pyttsx3")

llm_chat.append_tool(
    create_speak_tool(
        mode=VOICE_MODE,
        genai_tts_model="gemini-2.5-flash-preview-tts",  # Optional
        genai_voice_name="Sulafat",  # Optional
        openai_tts_model="tts-1",  # Optional
        openai_voice_name="alloy",  # Optional
        sample_rate_out=24000,  # Optional
    )
)
llm_chat.prepend_tool_policy(auto_approve("speak"))
```

`fetch_youtube_transcript` (extra `[youtube]`) is another tool in the same package: `llm_chat.append_tool(fetch_youtube_transcript)`.

### pyttsx3 voice quality

pyttsx3 uses your system's TTS engine. On Linux, it uses espeak/espeak-ng.

1. **Install espeak-ng for better voices**:
   ```bash
   # Ubuntu/Debian
   sudo apt install espeak-ng
   
   # Fedora
   sudo dnf install espeak-ng
   ```

2. **List available voices**:
   ```python
   from zrb_extras.llm.tool.pyttsx3.speak import list_available_voices
   for voice in list_available_voices():
       print(f"{voice['id']}: {voice['name']}")
   ```

3. **Configure voice via environment variables**:
   ```bash
   # Set a specific voice (espeak-ng variants)
   export PYTTSX3_VOICE_NAME="english-us+m3"   # Male voice
   # export PYTTSX3_VOICE_NAME="english-us+f3" # Female voice
   
   # Adjust speed (words per minute, default 150)
   export PYTTSX3_VOICE_RATE="150"
   
   # Adjust volume (0.0 to 1.0, default 1.0)
   export PYTTSX3_VOICE_VOLUME="0.9"
   ```

4. **Or pass to create_speak_tool**:
   ```python
   speak_tool = create_speak_tool(
       mode="pyttsx3",
       pyttsx3_voice_name="english-us+m3",  # Specific voice
       pyttsx3_rate=150,                     # Words per minute
       pyttsx3_volume=0.9,                   # Volume (0.0-1.0)
   )
   ```

### macOS Users

On macOS, pyttsx3 falls back to the native `say` command which has better quality. You can use any installed macOS voice:

```bash
# List available voices
say -v ?

# Set voice
export PYTTSX3_VOICE_NAME="Samantha"  # Female voice
# export PYTTSX3_VOICE_NAME="Daniel"  # Male voice
```
```

# For maintainers

## Publish to pypi

To publish zrb-extras, you need to have a `Pypi` account:

- Log in or register to [https://pypi.org/](https://pypi.org/)
- Create an API token

You can also create a `TestPypi` account:

- Log in or register to [https://test.pypi.org/](https://test.pypi.org/)
- Create an API token

Once you have your API token, you need to configure poetry:

```
poetry config pypi-token.pypi <your-api-token>
```

To publish zrb-extras, you can do the following command:

```bash
poetry publish --build
```

## Updating version

You can update zrb-extras version by modifying the following section in `pyproject.toml`:

```toml
[project]
version = "0.0.2"
```

## Adding dependencies

To add zrb-extras dependencies, you can edit the following section in `pyproject.toml`:

```toml
[project]
dependencies = [
    "Jinja2==3.1.2",
    "jsons==1.6.3"
]
```

## Adding script

To make zrb-extras executable, you can edit the following section in `pyproject.toml`:

```toml
[project-scripts]
zrb-extras-hello = "zrb_extras.__main__:hello"
```

Now, whenever you run `zrb-extras-hello`, the `main` function on your `__main__.py` will be executed.
