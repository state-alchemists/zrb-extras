import asyncio
import collections
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, AsyncGenerator, Deque, List, Optional

if TYPE_CHECKING:
    import numpy as np

# Constants
SPEECH_THRESHOLD_MULTIPLIER = 2.0
PRE_BUFFER_DURATION = 0.5  # seconds
CALIBRATION_DURATION = 0.5  # seconds
CALIBRATION_MAX_BLOCKS = 10
SPEECH_TIMEOUT = 30.0  # seconds
ADAPTATION_RATE = 0.05
RETENTION_RATE = 0.95


class VADState:
    """
    Encapsulates the Voice Activity Detection state and adaptive thresholding logic.
    """

    def __init__(
        self,
        initial_noise_level: float,
        min_threshold: float,
        multiplier: float = SPEECH_THRESHOLD_MULTIPLIER,
    ):
        self.noise_level = initial_noise_level
        self.min_threshold = min_threshold
        self.multiplier = multiplier
        self.threshold = self._calculate_threshold()
        self.speech_started = False
        self.silence_start_time: Optional[float] = None

    def _calculate_threshold(self) -> float:
        return max(self.noise_level * self.multiplier, self.min_threshold)

    def update_background_noise(self, energy: float):
        """Updates noise level and threshold if energy is low."""
        if energy < self.threshold:
            self.noise_level = (
                RETENTION_RATE * self.noise_level + ADAPTATION_RATE * energy
            )
            self.threshold = self._calculate_threshold()

    def is_speech(self, energy: float) -> bool:
        return energy > self.threshold

    def is_silence_timeout(self, energy: float, max_silence: float) -> bool:
        """
        Checks if silence has persisted longer than max_silence.
        Returns True if timeout is reached.
        """
        if energy < self.threshold:
            if self.silence_start_time is None:
                self.silence_start_time = time.time()
            elif time.time() - self.silence_start_time > max_silence:
                return True
        else:
            self.silence_start_time = None
        return False


async def record_until_silence(
    sample_rate: int,
    channels: int,
    silence_threshold: float,
    max_silence: float,
) -> "np.ndarray":
    """
    Records audio from the microphone until silence is detected.
    Uses an adaptive energy threshold to ignore background noise.
    """
    try:
        import numpy as np
    except ImportError:
        raise ImportError(
            "numpy is not installed. Please install zrb-extras[all] or specific extras."
        )

    async with _audio_stream(sample_rate, channels) as queue:
        # 1. Calibration Phase
        calibration_frames = await _calibrate_noise(queue)

        # Calculate initial noise statistics
        if calibration_frames:
            energies = [np.linalg.norm(b) / len(b) for b in calibration_frames]
            initial_noise_level = float(np.mean(energies))
        else:
            initial_noise_level = 0.0

        # Initialize VAD State
        vad_state = VADState(initial_noise_level, silence_threshold)

        print(
            f"Noise level: {vad_state.noise_level:.6f}, "
            f"Threshold: {vad_state.threshold:.6f}",
        )
        print("Waiting for speech...")

        # 2. Recording Loop
        return await _record_loop(
            queue,
            vad_state,
            sample_rate,
            max_silence,
            initial_buffer=calibration_frames,
        )


@asynccontextmanager
async def _audio_stream(
    sample_rate: int, channels: int
) -> AsyncGenerator[asyncio.Queue, None]:
    """
    Context manager that yields an asyncio.Queue receiving audio blocks from sounddevice.
    """
    try:
        import sounddevice as sd
    except ImportError:
        raise ImportError(
            "sounddevice is not installed. Please install zrb-extras[all] or specific extras."
        )

    loop = asyncio.get_running_loop()
    q = asyncio.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"Sounddevice status: {status}")
        # Use loop.call_soon_threadsafe to interact with asyncio queue from non-async thread
        loop.call_soon_threadsafe(q.put_nowait, indata.copy())

    print("Initializing microphone...")
    try:
        with sd.InputStream(
            samplerate=sample_rate, channels=channels, callback=callback
        ):
            yield q
    except Exception as e:
        print(f"Microphone initialization failed: {e}")
        raise


async def _calibrate_noise(queue: asyncio.Queue) -> List["np.ndarray"]:
    """
    Collects audio blocks for a short duration to estimate background noise.
    """
    calibration_frames = []
    start_time = time.time()

    while (
        time.time() - start_time < CALIBRATION_DURATION
        and len(calibration_frames) < CALIBRATION_MAX_BLOCKS
    ):
        try:
            block = await asyncio.wait_for(queue.get(), timeout=1.0)
            calibration_frames.append(block)
        except asyncio.TimeoutError:
            break

    return calibration_frames


async def _record_loop(
    queue: asyncio.Queue,
    vad_state: VADState,
    sample_rate: int,
    max_silence: float,
    initial_buffer: List["np.ndarray"],
) -> "np.ndarray":
    """
    Main loop that waits for speech, records it, and stops after silence.
    """
    import numpy as np

    rec_data: List[np.ndarray] = []
    pre_buffer: Deque[np.ndarray] = collections.deque()

    # Pre-populate pre-buffer with calibration frames to avoid losing immediate speech
    # (though typically calibration is noise-only, it's safer to keep)
    if initial_buffer:
        pre_buffer.extend(initial_buffer)

    while True:
        try:
            # Wait for audio block
            block = await asyncio.wait_for(queue.get(), timeout=SPEECH_TIMEOUT)
        except asyncio.TimeoutError:
            msg = (
                "Recording timed out."
                if vad_state.speech_started
                else "No speech detected."
            )
            print(msg)
            break

        energy = np.linalg.norm(block) / len(block)

        if not vad_state.speech_started:
            # --- Waiting for Speech ---
            _update_pre_buffer(pre_buffer, block, sample_rate)
            vad_state.update_background_noise(energy)

            if vad_state.is_speech(energy):
                print("Speech detected!")
                vad_state.speech_started = True
                rec_data.extend(pre_buffer)
        else:
            # --- Recording Speech ---
            rec_data.append(block)

            if vad_state.is_silence_timeout(energy, max_silence):
                print("Silence detected.")
                break

    if not rec_data:
        return np.array([], dtype=np.int16)

    # Combine and convert to PCM 16-bit
    audio_data = np.concatenate(rec_data, axis=0)
    audio_data = (audio_data * 32767).astype(np.int16)
    return audio_data


def _update_pre_buffer(
    buffer: Deque["np.ndarray"], block: "np.ndarray", sample_rate: int
):
    """
    Appends block to buffer and maintains the maximum duration.
    """
    buffer.append(block)
    # Estimate max buffer size based on current block length
    # (assuming roughly constant block size)
    if len(block) > 0:
        max_len = int(PRE_BUFFER_DURATION * sample_rate / len(block))
        while len(buffer) > max_len:
            buffer.popleft()
