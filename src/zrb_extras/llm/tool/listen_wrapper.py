from typing import TYPE_CHECKING, Any, Callable, Coroutine

from zrb_extras.llm.tool.sound_classifier import (
    SoundClassification,
    create_sound_classifier,
)

if TYPE_CHECKING:
    from pydantic_ai.models import Model
    from pydantic_ai.settings import ModelSettings
    from zrb.llm.config.limiter import LLMLimiter as LLMRateLimitter


def create_listen_tool_with_classification(
    base_listen_tool: Callable[[], Coroutine[Any, Any, str]],
    use_sound_classifier: bool = False,
    # Classification parameters
    classification_model: "Model | str | None" = None,
    classification_model_settings: "ModelSettings | None" = None,
    classification_system_prompt: str | None = None,
    classification_retries: int = 2,
    rate_limitter: "LLMRateLimitter | None" = None,
    # Behavior parameters
    fail_safe: bool = True,
) -> Callable[[], Coroutine[Any, Any, str]]:
    """
    Wraps a listen tool with optional sound classification.

    Args:
        base_listen_tool: The underlying listen tool to wrap.
        use_sound_classifier: Whether to enable sound classification.
        classification_model: Model to use for classification.
        classification_model_settings: Settings for the classification model.
        classification_system_prompt: System prompt for the classifier.
        classification_retries: Number of retries for classification.
        rate_limitter: Rate limiter for LLM calls.
        fail_safe: If True, classifier failures default to handling speech.

    Returns:
        A wrapped listen tool that optionally classifies transcripts.
    """

    async def listen_with_classification() -> str:
        """
        Listens for speech, transcribes it, and optionally classifies the result.

        Note: VAD is always used by the underlying listen tool for initial detection.
        """
        # Step 1: Use the base listen tool (which always uses VAD)
        transcript = await base_listen_tool()
        # Step 2: If classification is disabled, return transcript directly
        if not use_sound_classifier:
            return transcript
        # Step 3: Create classifier and classify transcript
        classify_transcript = create_sound_classifier(
            rate_limitter=rate_limitter,
            classification_model=classification_model,
            classification_model_settings=classification_model_settings,
            classification_system_prompt=classification_system_prompt,
            classification_retries=classification_retries,
        )
        classification: SoundClassification = await classify_transcript(transcript)
        # Step 4: Decide based on classification result
        return transcript if classification.get("is_speech", False) else ""

    # Preserve the original tool name and description if possible
    if hasattr(base_listen_tool, "__name__"):
        listen_with_classification.__name__ = f"{base_listen_tool.__name__}"
    if hasattr(base_listen_tool, "__doc__"):
        listen_with_classification.__doc__ = base_listen_tool.__doc__

    return listen_with_classification
