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
        # Sound Classification (optional)
        use_sound_classifier=True,  # Enable sound classification
        classification_model=None,  # Use default small model
        classification_system_prompt="Classify if the transcript contains actual speech or just background noise/fillers",
        classification_retries=2,  # Retry classification on failure
        fail_safe=True,  # Default to handling as speech if classification fails
    )
)
```


## Sound Classification Feature

The `create_listen_tool` now includes an optional sound classification feature that uses an LLM to analyze transcripts and determine if they contain actual speech or just background noise, fillers, or non-speech sounds.

### Key Features:

1. **VAD is always used** for initial speech detection (already implemented in existing listen tools)
2. **When `use_sound_classifier=True`**, transcripts are classified by an LLM using zrb's small model configuration system
3. **Fail-safe default**: If the classifier fails, it assumes the sound should be handled as speech
4. **Structured output**: Uses structured output types similar to `../zrb/src/zrb/task/llm/history_processor.py` pattern
5. **Configurable**: Supports custom models, prompts, retries, and rate limiting

### Usage Examples:

```python
# Basic usage with sound classification
listen_tool = create_listen_tool(
    mode="vosk",
    use_sound_classifier=True,
    tool_name="smart_listen"
)

# With custom classification settings
listen_tool = create_listen_tool(
    mode="google",
    use_sound_classifier=True,
    classification_model="custom-model",
    classification_model_settings={"temperature": 0.1},
    classification_system_prompt="Classify speech vs noise",
    classification_retries=3,
    fail_safe=False,  # Raise exception on classification failure
    rate_limitter=my_rate_limiter,
    tool_name="custom_classifier_listen"
)

# Backward compatibility - old code still works
listen_tool = create_listen_tool(
    mode="termux",
    # No use_sound_classifier parameter
    tool_name="basic_listen"
)
```

### How It Works:

1. The underlying listen tool (Vosk, Google, OpenAI, or Termux) captures audio and transcribes it
2. VAD (Voice Activity Detection) filters out silent periods
3. If `use_sound_classifier=True`, the transcript is sent to an LLM classifier
4. The classifier returns a structured response indicating:
   - `is_speech`: Boolean indicating if it's actual speech
   - `confidence`: Confidence score (0.0 to 1.0)
   - `category`: Optional category (e.g., "speech", "noise", "filler")
5. Based on the classification:
   - If `is_speech=True`: Returns the transcript
   - If `is_speech=False`: Returns empty string (ignores non-speech)

### Benefits:

- **Reduces false positives**: Filters out background noise, coughs, throat clearing, etc.
- **Improves accuracy**: Only processes actual speech content
- **Configurable**: Can be tuned for different environments and use cases
- **Backward compatible**: Existing code continues to work without changes

## Improving Voice Quality (Vosk Mode)

When using `VOICE_MODE=vosk`, speech recognition uses offline Vosk models and text-to-speech uses pyttsx3. Here's how to improve quality:

### Vosk Speech Recognition Models

**Recommended:** For best accuracy, use the larger model. The default small model (~40MB) has limited accuracy.

| Model | Size | Accuracy | Recommended |
|-------|------|----------|-------------|
| `vosk-model-en-us-0.22` | ~1.8GB | Best | ✅ **Yes** |
| `vosk-model-en-us-daanzu-20200905` | ~1GB | Good | Good balance |
| `vosk-model-small-en-us-0.15` | ~40MB | Limited | Default (not recommended) |

**Easiest way: Auto-download (recommended)**

Vosk auto-downloads models to `~/.cache/vosk/` when you specify `model_name`. Just configure it in `zrb_init.py`:

```python
listen = create_listen_tool(
    mode="vosk",
    vosk_model_name="vosk-model-en-us-0.22",  # Auto-downloads on first use
    # ... other options
)
```

**Alternative: Manual download**

If you prefer to pre-download (e.g., on a machine with better internet):
```bash
mkdir -p ~/.cache/vosk
cd ~/.cache/vosk
wget https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip
unzip vosk-model-en-us-0.22.zip
rm vosk-model-en-us-0.22.zip
```

**Alternative: Use `vosk_model_path` for custom locations:**
```python
listen_tool = create_listen_tool(
    mode="vosk",
    vosk_model_path="/custom/path/to/vosk-model-en-us-0.22",
)
```

### pyttsx3 Text-to-Speech Quality

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
       mode="vosk",
       voice_name="english-us+m3",  # Specific voice
       rate=150,                     # Words per minute
       volume=0.9,                   # Volume (0.0-1.0)
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
