# Zrb extras

zrb-extras is a [pypi](https://pypi.org) package.

You can install zrb-extras by invoking the following command:

```bash
pip install zrb-extras
pip install "zrb-extras[pyttsx3]"  # or [youtube], [all]
```

## Let your LLM task `speak`

`create_speak_tool` gives the agent a tool that says text aloud. The agent decides what to say and when, which suits an `LLMTask` that runs unattended ("the build finished") or a persona that should talk rather than print.

Talking *to* zrb is built into zrb itself since 3.10: `/voice` and `/handsfree` dictate, and hands-free can answer tool approvals. zrb can also read every reply aloud (`ZRB_LLM_SPEECH_ENABLED=on`). See zrb's [Voice and Camera](https://github.com/state-alchemists/zrb/blob/main/docs/llm/voice-camera.md) guide. The `listen` tool that used to live here was removed in zrb-extras 3.1.0 for that reason, and since 4.0.0 `speak` runs on zrb's speech backends instead of its own OpenAI, Gemini, Termux and pyttsx3 clients.

> Use the `speak` tool **or** zrb's built-in speech, not both, or replies are said twice.

### Or: a pyttsx3 voice for zrb's built-in speech

zrb reads replies aloud with Termux:API (on Android), `say`, `espeak-ng`, OpenAI or Gemini. zrb-extras adds `Pyttsx3SpeechBackend`: offline, in process, extra `[pyttsx3]`. If it fails, zrb's local engine speaks instead.

```python
from zrb.builtin import llm_chat
from zrb.llm.speech import SpeechConfig, enable_speech
from zrb_extras.llm.speech import Pyttsx3SpeechBackend

enable_speech(
    llm_chat,
    SpeechConfig(backend=Pyttsx3SpeechBackend(rate=150), enabled=True),
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

`create_speak_tool` takes zrb's `SpeechConfig`, so it speaks through the same backends as zrb's built-in speech: `auto` (Termux:API, `say` or `espeak-ng`), `openai`, `gemini`, a Pipecat voice like `kokoro` (needs `zrb[voice]`), or a backend object such as `Pyttsx3SpeechBackend`. Fields left out are read from `ZRB_LLM_SPEECH_*`.

```python
from zrb.builtin import llm_chat
from zrb.llm.speech import SpeechConfig
from zrb.llm.tool_call.tool_policy.auto_approve import auto_approve
from zrb_extras.llm.tool import create_speak_tool

llm_chat.append_tool(
    create_speak_tool(SpeechConfig(backend="openai", voice="alloy"))
)
llm_chat.prepend_tool_policy(auto_approve("speak"))
```

The agent can pick a `voice_name` and an `instruction` ("cheerful", "whisper") on each call. OpenAI's `gpt-4o-mini-tts` and Gemini follow the instruction; a backend passed as an object ignores both.

`fetch_youtube_transcript` (extra `[youtube]`) is another tool in the same package: `llm_chat.append_tool(fetch_youtube_transcript)`.

### pyttsx3 voices

pyttsx3 uses your system's TTS engine (espeak/espeak-ng on Linux; `sudo apt install espeak-ng` for better voices).

```python
from zrb_extras.llm.speech import Pyttsx3SpeechBackend
from zrb_extras.llm.speech.backend.pyttsx3 import list_available_voices

for voice in list_available_voices():
    print(f"{voice['id']}: {voice['name']}")

create_speak_tool(
    SpeechConfig(backend=Pyttsx3SpeechBackend(voice_name="english-us+m3", rate=150, volume=0.9))
)
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
