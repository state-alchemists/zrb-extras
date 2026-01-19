#!/usr/bin/env python3
"""
Integration tests for factory module with sound classification.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from zrb_extras.llm.tool.factory import create_listen_tool


class TestFactoryIntegration:
    """Test factory integration with sound classification."""

    @pytest.mark.asyncio
    async def test_create_listen_tool_without_classification(self):
        """Test creating listen tool without sound classification."""
        # Mock the underlying listen tool creation
        with patch("zrb_extras.llm.tool.factory.create_vosk_listen_tool") as mock_vosk:
            mock_tool = AsyncMock()
            mock_tool.return_value = "Test transcript"
            mock_vosk.return_value = mock_tool

            # Create listen tool without classification
            tool = create_listen_tool(
                mode="vosk",
                use_sound_classifier=False,  # Default is False
                tool_name="test_tool",
            )

            # Should return the basic tool
            assert tool == mock_tool

            # Should call vosk factory with correct params
            mock_vosk.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_listen_tool_with_classification(self):
        """Test creating listen tool with sound classification."""
        # Mock dependencies
        with patch(
            "zrb_extras.llm.tool.factory.create_vosk_listen_tool"
        ) as mock_vosk, patch(
            "zrb_extras.llm.tool.factory.create_listen_tool_with_classification"
        ) as mock_wrapper:

            # Setup mocks
            mock_basic_tool = AsyncMock()
            mock_vosk.return_value = mock_basic_tool

            mock_wrapped_tool = AsyncMock()
            mock_wrapped_tool.return_value = "Classified transcript"
            mock_wrapper.return_value = mock_wrapped_tool

            # Create listen tool WITH classification
            tool = create_listen_tool(
                mode="vosk",
                use_sound_classifier=True,
                classification_system_prompt="Classify speech",
                tool_name="test_tool_with_classification",
            )

            # Should return the wrapped tool
            assert tool == mock_wrapped_tool

            # Verify factory calls
            mock_vosk.assert_called_once()
            mock_wrapper.assert_called_once_with(
                base_listen_tool=mock_basic_tool,
                use_sound_classifier=True,
                classification_model=None,
                classification_model_settings=None,
                classification_system_prompt="Classify speech",
                classification_retries=2,
                rate_limitter=None,
                fail_safe=True,  # Default
            )

            # Test the tool
            result = await tool()
            assert result == "Classified transcript"

    @pytest.mark.asyncio
    async def test_create_listen_tool_with_custom_classification_settings(self):
        """Test creating listen tool with custom classification settings."""
        with patch(
            "zrb_extras.llm.tool.factory.create_vosk_listen_tool"
        ) as mock_vosk, patch(
            "zrb_extras.llm.tool.listen_wrapper.create_sound_classifier"
        ) as mock_classifier_factory:

            # Setup mocks
            mock_basic_tool = AsyncMock()
            mock_vosk.return_value = mock_basic_tool

            mock_classifier = AsyncMock()
            mock_classifier.return_value = {
                "is_speech": True,
                "confidence": 0.9,
                "category": "speech",
            }
            mock_classifier_factory.return_value = mock_classifier

            # Create listen tool with custom settings
            tool = create_listen_tool(
                mode="vosk",
                use_sound_classifier=True,
                classification_model="custom-model",
                classification_model_settings={"temperature": 0.1},
                classification_system_prompt="Custom prompt",
                classification_retries=3,
                fail_safe=False,
                tool_name="test_custom",
            )
            # Mock the base tool to return a transcript
            mock_basic_tool.return_value = "Test transcript"

            # Call the tool
            result = await tool()

            # Should return the transcript (since classifier says it's speech)
            assert result == "Test transcript"

            # Verify classifier factory was called with custom settings
            mock_classifier_factory.assert_called_once()
            call_kwargs = mock_classifier_factory.call_args[1]

            # Should pass custom settings to classifier factory
            assert call_kwargs["classification_model"] == "custom-model"
            assert call_kwargs["classification_model_settings"] == {"temperature": 0.1}
            assert call_kwargs["classification_system_prompt"] == "Custom prompt"
            assert call_kwargs["classification_retries"] == 3

            # Verify classifier was called
            mock_classifier.assert_called_once_with("Test transcript")

    @pytest.mark.asyncio
    async def test_create_listen_tool_backward_compatibility(self):
        """Test backward compatibility - old code should still work."""
        with patch("zrb_extras.llm.tool.factory.create_vosk_listen_tool") as mock_vosk:
            mock_tool = AsyncMock()
            mock_vosk.return_value = mock_tool

            # Old-style call without use_sound_classifier parameter
            tool = create_listen_tool(
                mode="vosk",
                tool_name="old_tool",
                # No use_sound_classifier parameter
            )

            # Should work and return basic tool
            assert tool == mock_tool
            mock_vosk.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_listen_tool_different_modes(self):
        """Test that different modes work with classification."""
        modes = ["vosk", "google", "openai", "termux"]

        for mode in modes:
            with patch(
                f"zrb_extras.llm.tool.factory.create_{mode}_listen_tool"
            ) as mock_factory, patch(
                "zrb_extras.llm.tool.listen_wrapper.create_sound_classifier"
            ) as mock_classifier_factory:

                # Reset mocks for each iteration
                mock_factory.reset_mock()
                mock_classifier_factory.reset_mock()

                # Setup mocks
                mock_basic_tool = AsyncMock()
                mock_basic_tool.return_value = f"Test transcript from {mode}"
                mock_factory.return_value = mock_basic_tool

                mock_classifier = AsyncMock()
                mock_classifier.return_value = {
                    "is_speech": True,
                    "confidence": 0.8,
                    "category": "speech",
                }
                mock_classifier_factory.return_value = mock_classifier

                # Create tool with classification for this mode
                tool = create_listen_tool(
                    mode=mode,
                    use_sound_classifier=True,
                    tool_name=f"test_{mode}",
                )

                # Should call the correct factory
                mock_factory.assert_called_once()

                # Call the tool
                result = await tool()

                # Should return the transcript (since classifier says it's speech)
                assert result == f"Test transcript from {mode}"

                # Verify classifier factory was called
                mock_classifier_factory.assert_called_once()

                # Verify classifier was called with the transcript
                mock_classifier.assert_called_once_with(f"Test transcript from {mode}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
