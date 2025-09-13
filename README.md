# Zrb extras

zrb-extras is a [pypi](https://pypi.org) package.

You can install zrb-extras by invoking the following command:

```bash
pip install zrb-extras
```

## Let your `LLMTask` `speak` and `listen`

### Prerequisites

Ubuntu

```bash
sudo apt install libasound2-dev portaudio19-dev pulseaudio
```

Termux

> First of all, make sure termux has permission to access microphone/speaker

```bash
pkg update && pkg upgrade -y
pkg install pulseaudio termux-api -y
# start PulseAudio daemon
pulseaudio --start --exit-idle-time=-1
# allow local TCP connections from guest
pactl load-module module-native-protocol-tcp auth-ip-acl=127.0.0.1 auth-anonymous=1
pactl load-module module-sles-source source_name=termux_mic latency_msec=60
```

Proot-distro (Ubuntu)

```bash
sudo apt install libasound2-dev portaudio19-dev pulseaudio
export PULSE_SERVER=tcp:127.0.0.1
export PULSE_SOURCE=termux_mic
```

### Create `zrb_init.py`

```python
from zrb.builtin import llm_ask
from zrb import llm_config
from zrb_extras.llm.tool import create_listen_tool, create_speak_tool

API_KEY = os.getenv("GOOGLE_API_KEY", "")

llm_ask.add_tool(
    create_speak_tool(
        api_key=API_KEY,  # Optional, by default taken from GEMINI_API_KEY or GOOOGLE_API_KEY
        stt_model="gemini-2.5-flash-preview-tts",  # Optional
        voice_name="Sulafat",  # Optional (https://ai.google.dev/gemini-api/docs/speech-generation#voices)
        sample_rate_out=24000,  # Optional
    )
)
llm_ask.add_tool(
    create_listen_tool(
        api_key=API_KEY,  # Optional, by default taken from GEMINI_API_KEY or GOOOGLE_API_KEY
        tts_model="gemini-2.5-flash",  # Optional
        sample_rate=16000,  # Optional
        channels=1,  # Optional
        silence_threshold=0.01,  # Optional (smaller means more sensitive)
        max_silence=4.0,  # Optional (4 second silence before stop listening)
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
