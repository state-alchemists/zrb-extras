#!/usr/bin/env python3
"""
Unit tests for listen wrapper module.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from zrb_extras.llm.tool.listen_wrapper import (
    create_listen_tool_with_classification,
)


class TestListenWrapper:
    """Test listen wrapper functionality."""

    @pytest.mark.asyncio
    async def test_create_listen_tool_with_classification_basic(self):
        """Test basic wrapper creation."""
        # Create mock listen tool
        mock_listen_tool = AsyncMock()
        mock_listen_tool.return_value = "Mock transcript"

        # Create mock classifier function
        mock_classifier = AsyncMock()
        mock_classifier.return_value = {
            "is_speech": True,
            "confidence": 0.95,
            "category": "speech",
            "reason": "Clear speech",
        }

        # Create wrapper with mocked create_sound_classifier
        with patch(
            "zrb_extras.llm.tool.listen_wrapper.create_sound_classifier"
        ) as mock_create_classifier:
            mock_create_classifier.return_value = mock_classifier

            wrapped_tool = create_listen_tool_with_classification(
                base_listen_tool=mock_listen_tool,
                use_sound_classifier=True,
                fail_safe=True,
            )

            # Should return a function
            assert callable(wrapped_tool)

            # Call wrapped tool
            result = await wrapped_tool()

            # Should return the transcript
            assert result == "Mock transcript"

            # Verify calls
            mock_listen_tool.assert_called_once_with()
            mock_create_classifier.assert_called_once()
            mock_classifier.assert_called_once_with("Mock transcript")

    @pytest.mark.asyncio
    async def test_create_listen_tool_with_classification_not_speech(self):
        """Test wrapper when classifier returns not speech."""
        # Create mock listen tool
        mock_listen_tool = AsyncMock()
        mock_listen_tool.return_value = "Mock transcript"

        # Create mock classifier that returns not speech
        mock_classifier = AsyncMock()
        mock_classifier.return_value = {
            "is_speech": False,
            "confidence": 0.1,
            "category": "noise",
            "reason": "Background noise",
        }

        # Create wrapper with mocked create_sound_classifier
        with patch(
            "zrb_extras.llm.tool.listen_wrapper.create_sound_classifier"
        ) as mock_create_classifier:
            mock_create_classifier.return_value = mock_classifier

            wrapped_tool = create_listen_tool_with_classification(
                base_listen_tool=mock_listen_tool,
                use_sound_classifier=True,
                fail_safe=True,
            )

            # Call wrapped tool
            result = await wrapped_tool()

            # Should return empty string when not speech
            assert result == ""

    @pytest.mark.asyncio
    async def test_create_listen_tool_with_classification_empty_transcript(self):
        """Test wrapper with empty transcript."""
        # Create mock listen tool that returns empty transcript
        mock_listen_tool = AsyncMock()
        mock_listen_tool.return_value = ""

        # Create mock classifier that returns classification for empty transcript
        mock_classifier = AsyncMock()
        mock_classifier.return_value = {
            "is_speech": False,
            "confidence": 1.0,
            "category": "empty",
            "reason": "Empty transcript",
        }

        # Create wrapper with mocked create_sound_classifier
        with patch(
            "zrb_extras.llm.tool.listen_wrapper.create_sound_classifier"
        ) as mock_create_classifier:
            mock_create_classifier.return_value = mock_classifier

            wrapped_tool = create_listen_tool_with_classification(
                base_listen_tool=mock_listen_tool,
                use_sound_classifier=True,
                fail_safe=True,
            )

            # Call wrapped tool
            result = await wrapped_tool()

            # Should return empty string (not speech)
            assert result == ""

            # Classifier SHOULD be called even for empty transcript
            mock_classifier.assert_called_once_with("")

    @pytest.mark.asyncio
    async def test_create_listen_tool_with_classification_classifier_error(self):
        """Test wrapper when classifier raises error."""
        # Create mock listen tool
        mock_listen_tool = AsyncMock()
        mock_listen_tool.return_value = "Mock transcript"

        # Create mock classifier that returns fail-safe classification
        mock_classifier = AsyncMock()
        mock_classifier.return_value = {
            "is_speech": True,  # fail-safe default
            "confidence": 0.5,
            "category": "unknown",
            "reason": "Classification failed, defaulting to safe mode",
        }

        # Create wrapper with fail_safe=True and mocked create_sound_classifier
        with patch(
            "zrb_extras.llm.tool.listen_wrapper.create_sound_classifier"
        ) as mock_create_classifier:
            mock_create_classifier.return_value = mock_classifier

            wrapped_tool = create_listen_tool_with_classification(
                base_listen_tool=mock_listen_tool,
                use_sound_classifier=True,
                fail_safe=True,
            )

            # Call wrapped tool
            result = await wrapped_tool()

            # Should return transcript (fail-safe default is speech)
            assert result == "Mock transcript"

            # Classifier should be called
            mock_classifier.assert_called_once_with("Mock transcript")

    @pytest.mark.asyncio
    async def test_create_listen_tool_with_classification_classifier_error_no_failsafe(
        self,
    ):
        """Test wrapper when classifier returns non-speech classification."""
        # Create mock listen tool
        mock_listen_tool = AsyncMock()
        mock_listen_tool.return_value = "Mock transcript"

        # Create mock classifier that returns non-speech classification
        mock_classifier = AsyncMock()
        mock_classifier.return_value = {
            "is_speech": False,
            "confidence": 0.8,
            "category": "noise",
            "reason": "Background noise",
        }

        # Create wrapper with mocked create_sound_classifier
        with patch(
            "zrb_extras.llm.tool.listen_wrapper.create_sound_classifier"
        ) as mock_create_classifier:
            mock_create_classifier.return_value = mock_classifier

            wrapped_tool = create_listen_tool_with_classification(
                base_listen_tool=mock_listen_tool,
                use_sound_classifier=True,
                fail_safe=False,
            )

            # Call wrapped tool
            result = await wrapped_tool()

            # Should return empty string (not speech)
            assert result == ""

            # Classifier should be called
            mock_classifier.assert_called_once_with("Mock transcript")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
