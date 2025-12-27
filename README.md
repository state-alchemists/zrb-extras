# Zrb extras

zrb-extras is a [pypi](https://pypi.org) package.

You can install zrb-extras by invoking the following command:

```bash
pip install zrb-extras
```

## Let your `LLMTask` `speak` and `listen`

### Prerequisites

#### Termux

> First of all, make sure termux has permission to access microphone/speaker

```bash
pkg update && pkg upgrade -y
pkg install pulseaudio termux-api -y
```

Run the following script or add it to `~/.bashrc`

```bash
# start PulseAudio daemon
pulseaudio --start --load="module-native-protocol-tcp auth-ip-acl=127.0.0.1 auth-anonymous=1" --exit-idle-time=-1

# load module now (if it errors, check you gave Termux:API mic permission and restart Termux)
pactl load-module module-sles-source
# confirm source exists
pactl list short sources

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
from zrb.builtin import llm_ask
from zrb import llm_config
from zrb_extras.llm.tool import create_listen_tool, create_speak_tool

# Valid modes: "google", "openai", "termux", "vosk"
VOICE_MODE = os.getenv("VOICE_MODE", "vosk")
if VOICE_MODE not in ("google", "openai", "termux", "vosk"):
    VOICE_MODE = "vosk"

llm_ask.add_tool(
    create_speak_tool(
        mode=VOICE_MODE,
        genai_tts_model="gemini-2.5-flash-preview-tts",  # Optional
        genai_voice_name="Sulafat",  # Optional
        openai_tts_model="tts-1",  # Optional
        openai_voice_name="alloy",  # Optional
        sample_rate_out=24000,  # Optional
    )
)
llm_ask.add_tool(
    create_listen_tool(
        mode=VOICE_MODE,
        genai_stt_model="gemini-2.5-flash",  # Optional
        openai_stt_model="whisper-1",  # Optional
        sample_rate=16000,  # Optional
        channels=1,  # Optional
        silence_threshold=0.01,  # Optional
        max_silence=4.0,  # Optional
    )
)

# Optional: allow LLM to speak or listen without asking for user approval
if not llm_config.default_yolo_mode:
    llm_config.set_default_yolo_mode(["speak", "listen"])
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
