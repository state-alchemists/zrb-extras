import json
import sys
import traceback
from typing import TYPE_CHECKING, Callable, Coroutine

from zrb import AnyContext
from zrb.config.llm_config import llm_config
from zrb.config.llm_rate_limitter import (
    LLMRateLimitter,
)
from zrb.config.llm_rate_limitter import llm_rate_limitter as default_llm_rate_limitter
from zrb.task.llm.agent_runner import run_agent_iteration
from zrb.util.cli.style import stylize_faint

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
    ctx: AnyContext,
    rate_limitter: LLMRateLimitter | None = None,
    classification_model: "Model | str | None" = None,
    classification_model_settings: "ModelSettings | None" = None,
    classification_system_prompt: str | None = None,
    classification_retries: int = 2,
) -> Callable[[str], Coroutine[None, None, SoundClassification]]:
    """
    Creates a sound classification function that uses LLM to classify transcripts.

    Args:
        ctx: The task context.
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
        classification_model = llm_config.default_small_model
    if classification_model_settings is None:
        classification_model_settings = llm_config.default_small_model_settings
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
            _print_info(ctx, "🔊 Classifying Sound", 2)
            classification_run = await run_agent_iteration(
                ctx=ctx,
                agent=classification_agent,
                user_prompt=classification_message,
                attachments=[],
                history_list=[],
                rate_limitter=rate_limitter,
                log_indent_level=2,
            )
            if (
                classification_run
                and classification_run.result
                and classification_run.result.output
            ):
                result = classification_run.result.output
                if isinstance(result, dict):
                    # Ensure required fields are present
                    is_speech = result.get("is_speech", True)  # Default to safe
                    confidence = result.get("confidence", 0.5)
                    category = result.get("category")
                    reason = result.get("reason")
                    # Log the final decision using _print_info
                    if is_speech:
                        _print_info(
                            ctx,
                            f"✅ Classified as USER REQUEST (confidence: {confidence:.2f})",
                            2,
                        )
                    else:
                        _print_info(
                            ctx,
                            f"❌ Classified as NOT USER REQUEST (category: {category}, confidence: {confidence:.2f})",  # noqa
                            2,
                        )
                    return SoundClassification(
                        is_speech=is_speech,
                        confidence=confidence,
                        category=category,
                        reason=reason,
                    )
        except BaseException as e:
            ctx.log_warning(f"Error during sound classification: {e}")
            traceback.print_exc()
        # Fallback: assume speech should be handled (fail-safe)
        ctx.log_warning(
            "Sound classification failed, assuming speech should be handled"
        )
        return SoundClassification(
            is_speech=True,
            confidence=0.5,
            category="unknown",
            reason="Classification failed, defaulting to safe mode",
        )

    return classify_transcript


def _print_info(ctx: AnyContext, text: str, log_indent_level: int = 0):
    log_prefix = (2 * (log_indent_level + 1)) * " "
    ctx.print(stylize_faint(f"{log_prefix}{text}"), plain=True)
