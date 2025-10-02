"""Utility client for interacting with OpenAI models via the Responses API."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from importlib import util as importlib_util
from typing import Optional, Tuple, List, Union


logger = logging.getLogger(__name__)

_OPENAI_AVAILABLE = importlib_util.find_spec("openai") is not None
_TIKTOKEN_AVAILABLE = importlib_util.find_spec("tiktoken") is not None

if _OPENAI_AVAILABLE:  # pragma: no cover - exercised only when dependency installed
    from openai import OpenAI  # type: ignore
else:  # pragma: no cover - handled gracefully by OpenAIClient
    OpenAI = None  # type: ignore

if _TIKTOKEN_AVAILABLE:  # pragma: no cover - optional dependency for token accounting
    import tiktoken  # type: ignore
else:  # pragma: no cover - handled gracefully via fallbacks
    tiktoken = None  # type: ignore


DEFAULT_CONTEXT_TOKEN_LIMIT = 128_000

# Best-effort context window sizes for common OpenAI models. Values represent the
# total token budget (prompt + completion) supported by the model.
_MODEL_CONTEXT_WINDOWS = {
    "gpt-4o-mini": 128_000,
    "gpt-4o": 128_000,
    "gpt-4.1": 128_000,
    "gpt-4.1-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4.1-preview": 128_000,
    "gpt-4.1-nano": 128_000,
    "gpt-3.5-turbo": 16_385,
}

# Notice inserted into prompts when truncation is required to stay within the
# model's context window. The message helps downstream debugging by signalling
# that some context was dropped before calling the OpenAI API.
_TRUNCATION_NOTICE = "\n\n[TRUNCATED: content shortened to fit token budget]\n\n"


class OpenAIIntegrationError(RuntimeError):
    """Raised when an OpenAI integration issue prevents a request."""


class OpenAIClient:
    """Lightweight wrapper around the OpenAI Python SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "gpt-4o-mini",
        max_context_tokens: Optional[int] = None,
    ) -> None:
        if not _OPENAI_AVAILABLE:
            raise OpenAIIntegrationError("The 'openai' package is not installed in this environment.")

        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise OpenAIIntegrationError("OpenAI API key is not configured. Set OPENAI_API_KEY to enable the integration.")

        self._client = OpenAI(api_key=resolved_key)
        self.default_model = os.getenv("OPENAI_MODEL", default_model)
        self.max_context_tokens = self._resolve_context_window(
            self.default_model,
            override=max_context_tokens or os.getenv("OPENAI_CONTEXT_WINDOW"),
        )

        safe_model = self.default_model or "<unspecified>"
        logger.debug("OpenAIClient initialized for model %s", safe_model)

    def generate_response(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        temperature: float = 0.6,
        max_output_tokens: int = 600,
    ) -> str:
        """Generate a response for the supplied prompt using OpenAI Responses API."""

        target_model = model or self.default_model
        if not target_model:
            raise OpenAIIntegrationError("No OpenAI model specified. Provide OPENAI_MODEL or pass model explicitly.")

        prompt, max_output_tokens = self._optimise_prompt_budget(
            prompt,
            target_model,
            max_output_tokens,
        )

        try:
            response = self._client.responses.create(
                model=target_model,
                input=prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
        except Exception as exc:  # pragma: no cover - network errors are environment specific
            raise OpenAIIntegrationError(
                f"Failed to generate response with OpenAI model '{target_model}': {exc}"
            ) from exc

        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text

        # Fall back to assembling text manually for SDK versions without output_text helper.
        output_chunks = getattr(response, "output", None)
        if not output_chunks:
            raise OpenAIIntegrationError("OpenAI response did not include any text output.")

        segments: list[str] = []
        for chunk in output_chunks:
            for content in getattr(chunk, "content", []) or []:
                text = getattr(content, "text", None)
                if isinstance(text, str):
                    segments.append(text)

        if not segments:
            raise OpenAIIntegrationError("OpenAI response contained no textual content.")

        return "".join(segments)

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Estimate the number of tokens for *text* under the requested model.

        When the optional :mod:`tiktoken` dependency is available the calculation
        is exact for supported models. Otherwise, a character-level approximation
        is used which still enables budget management without additional
        dependencies.
        """

        tokens, _ = self._encode_text(text, model)
        return len(tokens)

    def _optimise_prompt_budget(
        self,
        prompt: str,
        model: Optional[str],
        max_output_tokens: int,
    ) -> Tuple[str, int]:
        """Ensure the prompt and requested output fit within the model budget."""

        tokens, encoding = self._encode_text(prompt, model)
        input_tokens = len(tokens)
        if input_tokens + max_output_tokens <= self.max_context_tokens:
            return prompt, max_output_tokens

        allowed_input_tokens = max(self.max_context_tokens - max_output_tokens, 0)
        logger.info(
            "Prompt length %s tokens exceeds %s-token context window; reserving %s tokens for completion",
            input_tokens,
            self.max_context_tokens,
            max_output_tokens,
        )

        if allowed_input_tokens <= 0:
            truncated_tokens = tokens[: self.max_context_tokens]
            truncated_prompt = self._decode_tokens(truncated_tokens, encoding)
            logger.warning(
                "OpenAI prompt truncated to %s tokens leaving minimal room for completion; consider reducing prompt size",
                len(truncated_tokens),
            )
            return truncated_prompt, max(1, self.max_context_tokens - len(truncated_tokens))

        filler_tokens, _ = self._encode_text(_TRUNCATION_NOTICE, model, encoding)
        filler_len = len(filler_tokens)

        if allowed_input_tokens <= filler_len + 1:
            truncated_tokens = tokens[-allowed_input_tokens:]
            truncated_prompt = self._decode_tokens(truncated_tokens, encoding)
            logger.warning(
                "Prompt severely truncated to last %s tokens to satisfy context window.",
                len(truncated_tokens),
            )
            return truncated_prompt, max(1, self.max_context_tokens - len(truncated_tokens))

        base_budget = allowed_input_tokens - filler_len
        tail_tokens = min(len(tokens), max(0, base_budget // 3))
        head_tokens = max(0, base_budget - tail_tokens)

        truncated_sequence: List[Union[int, str]] = []
        if head_tokens:
            truncated_sequence.extend(tokens[:head_tokens])
        truncated_sequence.extend(filler_tokens)
        if tail_tokens:
            truncated_sequence.extend(tokens[-tail_tokens:])

        truncated_prompt = self._decode_tokens(truncated_sequence, encoding)
        input_budget = len(truncated_sequence)
        completion_budget = max(1, min(max_output_tokens, self.max_context_tokens - input_budget))

        logger.info(
            "Trimmed OpenAI prompt from %s to %s tokens; reserving %s tokens for completion.",
            input_tokens,
            input_budget,
            completion_budget,
        )

        return truncated_prompt, completion_budget

    def _encode_text(
        self,
        text: str,
        model: Optional[str],
        encoding=None,
    ) -> Tuple[List[Union[int, str]], Optional[object]]:
        """Encode *text* into token identifiers for the supplied model."""

        if encoding is None:
            encoding = self._get_encoding_for_model(model or self.default_model)

        if encoding is not None:
            return encoding.encode(text), encoding

        # Character-level fallback keeps behaviour deterministic even when the
        # optional tiktoken dependency is not installed.
        return list(text), None

    def _decode_tokens(
        self,
        tokens: List[Union[int, str]],
        encoding,
    ) -> str:
        if encoding is not None:
            return encoding.decode(tokens)
        return "".join(tokens)

    @staticmethod
    @lru_cache(maxsize=16)
    def _get_encoding_for_model(model_name: Optional[str]):
        if not model_name or not _TIKTOKEN_AVAILABLE or tiktoken is None:
            return None

        try:
            return tiktoken.encoding_for_model(model_name)
        except Exception:  # pragma: no cover - unexpected model identifiers
            try:
                return tiktoken.get_encoding("cl100k_base")
            except Exception:  # pragma: no cover - tiktoken misconfiguration
                return None

    def _resolve_context_window(self, model_name: Optional[str], override: Optional[object]) -> int:
        if isinstance(override, str) and override.strip():
            try:
                value = int(override)
                if value > 0:
                    return value
            except ValueError:
                logger.warning("Invalid OPENAI_CONTEXT_WINDOW value '%s'; falling back to defaults", override)
        elif isinstance(override, int) and override > 0:
            return override

        for known_model, window in _MODEL_CONTEXT_WINDOWS.items():
            if model_name and known_model in model_name:
                return window

        return DEFAULT_CONTEXT_TOKEN_LIMIT


def openai_sdk_available() -> bool:
    """Return True when the OpenAI dependency is importable."""

    return _OPENAI_AVAILABLE


__all__ = ["OpenAIClient", "OpenAIIntegrationError", "openai_sdk_available"]

