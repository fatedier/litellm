"""Tests for the Anthropic Messages to Responses API handler."""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../..")))

from litellm.llms.anthropic.experimental_pass_through.responses_adapters.handler import (
    _build_responses_kwargs,
)


class TestBuildResponsesKwargsPromptCacheKey:
    def _build(
        self, *, extra_kwargs: dict[str, object] | None = None, stream: bool | None = False
    ) -> dict[str, object]:
        return _build_responses_kwargs(
            max_tokens=1024,
            messages=[{"role": "user", "content": "hello"}],
            model="gpt-4o",
            stream=stream,
            extra_kwargs=extra_kwargs,
        )

    @pytest.mark.parametrize("stream", [False, True])
    def test_claude_code_session_header_becomes_prompt_cache_key(self, stream: bool):
        session_id = "b11d010a-7adf-45e2-8965-de0a738f598a"

        result = self._build(
            stream=stream,
            extra_kwargs={
                "proxy_server_request": {
                    "headers": {"X-Claude-Code-Session-Id": session_id},
                },
            },
        )

        assert result["prompt_cache_key"] == session_id

    def test_explicit_prompt_cache_key_is_preserved(self):
        session_id = "b11d010a-7adf-45e2-8965-de0a738f598a"
        explicit_key = "caller-defined-key"

        result = self._build(
            extra_kwargs={
                "prompt_cache_key": explicit_key,
                "proxy_server_request": {
                    "headers": {"x-claude-code-session-id": session_id},
                },
            },
        )

        assert result["prompt_cache_key"] == explicit_key

    def test_other_session_headers_do_not_create_prompt_cache_key(self):
        result = self._build(
            extra_kwargs={
                "litellm_session_id": "session-1234",
                "proxy_server_request": {
                    "headers": {"x-litellm-session-id": "session-1234"},
                },
            },
        )

        assert "prompt_cache_key" not in result

    def test_empty_session_header_does_not_create_prompt_cache_key(self):
        result = self._build(
            extra_kwargs={
                "proxy_server_request": {
                    "headers": {"x-claude-code-session-id": ""},
                },
            },
        )

        assert "prompt_cache_key" not in result
