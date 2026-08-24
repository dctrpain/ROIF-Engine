from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .expression_gate import ExpressionDecision
from .language_adapter import LanguageAdapter


@dataclass(frozen=True, slots=True)
class VerbalizationRequest:
    """
    Input passed to a speech backend.

    The backend receives only content already admitted by the
    ExpressionGate and rendered canonically by LanguageAdapter.
    """

    decision: ExpressionDecision
    canonical_text: str
    language: str = "ru"


@dataclass(frozen=True, slots=True)
class VerbalizationResult:
    """
    Result produced by the verbalization layer.
    """

    text: str
    model_name: str
    used_llm: bool


class LLMBackend(Protocol):
    """
    Minimal protocol for any future language backend.
    """

    def generate(
        self,
        request: VerbalizationRequest,
    ) -> str:
        ...


class CanonicalBackend:
    """
    Safe deterministic backend.

    It performs no generative rewriting and simply returns the
    canonical expression already produced from RDA evidence.
    """

    def generate(
        self,
        request: VerbalizationRequest,
    ) -> str:
        return request.canonical_text


class LLMVerbalizer:
    """
    Boundary between traceable RDA expression and optional
    natural-language generation.

    Fundamental invariant:

        LLMVerbalizer must not create the internal state.

    ExpressionDecision is formed before this layer.
    A future LLM backend may alter linguistic form only.
    """

    def __init__(
        self,
        backend: LLMBackend | None = None,
    ) -> None:
        self._backend = backend or CanonicalBackend()
        self._adapter = LanguageAdapter()

    def speak(
        self,
        decision: ExpressionDecision,
        *,
        language: str = "ru",
    ) -> VerbalizationResult:
        canonical = self._adapter.render(decision)

        request = VerbalizationRequest(
            decision=decision,
            canonical_text=canonical.text,
            language=language,
        )

        text = self._backend.generate(request)

        return VerbalizationResult(
            text=text,
            model_name=self._backend.__class__.__name__,
            used_llm=not isinstance(
                self._backend,
                CanonicalBackend,
            ),
        )
