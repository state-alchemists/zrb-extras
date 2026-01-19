import json
import sys
import traceback
from typing import TYPE_CHECKING, Callable, Coroutine

from zrb import llm_config, llm_limiter as default_llm_rate_limitter
from zrb.llm.config.limiter import LLMLimiter as LLMRateLimitter
from zrb.llm.agent import run_agent

if sys.version_info >= (3, 12):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

if TYPE_CHECKING:
    from pydantic_ai.models import Model
    from pydantic_ai.settings import ModelSettings


class SoundClassification(TypedDict):
    """
    Sound classification result

    Attributes:
        is_speech: Whether the sound is speech that should be handled
        confidence: Confidence level (0.0 to 1.0)
        category: Optional category of the sound (e.g., 'speech', 'noise', 'music')
        reason: Optional reason for classification
    """

    is_speech: bool
    confidence: float
    category: str | None
    reason: str | None


def classify_sound(sound_classification: SoundClassification):
    """
    Write sound classification result for decision making.
    """
    return sound_classification


def create_sound_classifier(
    rate_limitter: LLMRateLimitter | None = None,
    classification_model: "Model | str | None" = None,
    classification_model_settings: "ModelSettings | None" = None,
    classification_system_prompt: str | None = None,
    classification_retries: int = 2,
) -> Callable[[str], Coroutine[None, None, SoundClassification]]:
    """
    Creates a sound classification function that uses LLM to classify transcripts.

    Args:
        rate_limitter: Rate limiter for LLM calls.
        classification_model: Model to use for classification.
        classification_model_settings: Settings for the classification model.
        classification_system_prompt: System prompt for the classifier.
        classification_retries: Number of retries for classification.

    Returns:
        A coroutine function that takes a transcript and returns a SoundClassification.
    """
    from pydantic_ai import Agent

    if rate_limitter is None:
        rate_limitter = default_llm_rate_limitter
    if classification_model is None:
        classification_model = llm_config.model
    if classification_model_settings is None:
        classification_model_settings = llm_config.model_settings
    if classification_system_prompt is None:
        classification_system_prompt = (
            "You are a sound classifier. Analyze the provided transcript "
            "and determine if it contains speech that should be handled. "
            "Consider background noise, non-speech sounds, and unclear speech. "
            "If unsure, assume it's speech to be safe."
        )

    async def classify_transcript(transcript: str) -> SoundClassification:
        """
        Classifies a transcript to determine if it contains speech that should be handled.

        Args:
            transcript: The transcribed text to classify.

        Returns:
            SoundClassification dict with classification results.
            If classification fails, returns a safe default (is_speech=True).
        """
        if not transcript or transcript.strip() == "":
            # Empty transcript - assume not speech
            return SoundClassification(
                is_speech=False,
                confidence=1.0,
                category="empty",
                reason="Empty transcript",
            )
        classification_message = (
            f"Classify the following transcript: {json.dumps(transcript)}"
        )
        classification_agent = Agent[None, SoundClassification](
            model=classification_model,
            output_type=classify_sound,
            instructions=classification_system_prompt,
            model_settings=classification_model_settings,
            retries=classification_retries,
        )
        try:
            print("🔊 Classifying Sound")
            result, _ = await run_agent(
                agent=classification_agent,
                message=classification_message,
                message_history=[],
                limiter=default_llm_rate_limitter,
            )
            if isinstance(result, dict):
                # Ensure required fields are present
                is_speech = result.get("is_speech", True)  # Default to safe
                confidence = result.get("confidence", 0.5)
                category = result.get("category")
                reason = result.get("reason")
                if is_speech:
                    print(f"✅ Classified as USER REQUEST (confidence: {confidence:.2f})")
                else:
                    print(
                        f"❌ Classified as NOT USER REQUEST (category: {category}, confidence: {confidence:.2f})",  # noqa
                    )
                return SoundClassification(
                    is_speech=is_speech,
                    confidence=confidence,
                    category=category,
                    reason=reason,
                )
        except BaseException as e:
            print(f"Error during sound classification: {e}")
            traceback.print_exc()
        # Fallback: assume speech should be handled (fail-safe)
        print(
            "Sound classification failed, assuming speech should be handled"
        )
        return SoundClassification(
            is_speech=True,
            confidence=0.5,
            category="unknown",
            reason="Classification failed, defaulting to safe mode",
        )

    return classify_transcript
