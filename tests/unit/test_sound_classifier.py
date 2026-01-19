#!/usr/bin/env python3
"""
Unit tests for sound classifier module.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from zrb_extras.llm.tool.sound_classifier import (
    SoundClassification,
    classify_sound,
    create_sound_classifier,
)


class TestSoundClassification:
    """Test SoundClassification TypedDict."""

    def test_sound_classification_structure(self):
        """Test that SoundClassification has expected fields."""
        classification = SoundClassification(
            is_speech=True,
            confidence=0.95,
            category="speech",
            reason="Clear, intentional speech",
        )

        assert classification["is_speech"] is True
        assert classification["confidence"] == 0.95
        assert classification["category"] == "speech"
        assert classification["reason"] == "Clear, intentional speech"

    def test_sound_classification_optional_fields(self):
        """Test that SoundClassification optional fields work."""
        classification = SoundClassification(
            is_speech=False,
            confidence=0.3,
        )

        assert classification["is_speech"] is False
        assert classification["confidence"] == 0.3
        assert classification.get("category") is None
        assert classification.get("reason") is None


class TestClassifySound:
    """Test classify_sound output type function."""

    def test_classify_sound_signature(self):
        """Test that classify_sound has correct signature."""
        # classify_sound is an output type function that returns its input
        assert callable(classify_sound)

        # Test that it returns whatever is passed to it
        test_input = {
            "is_speech": True,
            "confidence": 0.95,
            "category": "speech",
            "reason": "Test",
        }
        result = classify_sound(test_input)
        assert result == test_input

    def test_classify_sound_identity(self):
        """Test that classify_sound acts as an identity function."""
        # Should return exactly what's passed in
        test_dict = {"is_speech": False, "confidence": 0.3}
        result = classify_sound(test_dict)
        assert result is test_dict  # Should be the same object


class TestCreateSoundClassifier:
    """Test create_sound_classifier factory function."""

    @pytest.mark.asyncio
    async def test_create_sound_classifier_basic(self):
        """Test basic sound classifier creation."""
        # Create classifier
        classifier = create_sound_classifier(
            classification_system_prompt="Classify transcripts",
        )

        # Should return a coroutine function
        assert callable(classifier)

    @pytest.mark.asyncio
    async def test_create_sound_classifier_with_mock_llm(self):
        """Test sound classifier with mocked LLM response."""
        # Mock LLM response
        mock_response = {
            "is_speech": True,
            "confidence": 0.9,
            "category": "speech",
            "reason": "Clear speech",
        }

        # Create a mock agent run result
        mock_run_result = MagicMock()
        mock_run_result.result = MagicMock()
        mock_run_result.result.output = mock_response
        mock_run_result.result.usage = MagicMock(return_value="tokens: 10")

        with patch(
            "zrb_extras.llm.tool.sound_classifier.run_agent_iteration"
        ) as mock_run_agent:
            mock_run_agent.return_value = mock_run_result

            # Create classifier
            classifier = create_sound_classifier(
                classification_system_prompt="Classify transcripts",
            )

            # Call classifier
            result = await classifier("Hello world")

            # Verify result
            assert result["is_speech"] is True
            assert result["confidence"] == 0.9
            assert result["category"] == "speech"
            assert result["reason"] == "Clear speech"

            # Verify LLM was called
            mock_run_agent.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_sound_classifier_empty_transcript(self):
        """Test sound classifier with empty transcript."""
        # Create classifier
        classifier = create_sound_classifier(
            classification_system_prompt="Classify transcripts",
        )

        # Call with empty transcript
        result = await classifier("")

        # Empty transcript should return False with confidence=1.0
        assert result["is_speech"] is False
        assert result["confidence"] == 1.0

    @pytest.mark.asyncio
    async def test_create_sound_classifier_llm_error(self):
        """Test sound classifier when LLM fails."""
        # Mock LLM to raise exception
        with patch(
            "zrb_extras.llm.tool.sound_classifier.run_agent_iteration"
        ) as mock_run_agent:
            mock_run_agent.side_effect = Exception("LLM error")

            # Create classifier
            classifier = create_sound_classifier(
                classification_system_prompt="Classify transcripts",
            )

            # Call classifier
            result = await classifier("Hello world")

            # Should return fail-safe default
            assert result["is_speech"] is True  # fail-safe default
            assert result["confidence"] == 0.5  # From the code
            assert result["category"] == "unknown"
            assert "Classification failed" in result["reason"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
