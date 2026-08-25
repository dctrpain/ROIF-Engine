from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .expression_gate import ExpressionDecision
from .expression_validator import (
    ExpressionValidator,
    ValidationStatus,
)
from .language_adapter import LanguageAdapter


@dataclass(frozen=True, slots=True)
class VerbalizationRequest:
    """
    Input passed to a speech backend.

    The backend receives only content already admitted by the
    ExpressionGate and rendered canonically by LanguageAdapter.

    Retry metadata may describe why a previous linguistic candidate
    was rejected, but it must never introduce new RDA evidence.
    """

    decision: ExpressionDecision
    canonical_text: str
    language: str = "ru"
    retry_reasons: tuple[str, ...] = ()
    previous_candidate: str | None = None

    @property
    def is_retry(self) -> bool:
        return bool(self.retry_reasons)


@dataclass(frozen=True, slots=True)
class VerbalizationResult:
    """
    Final outward result of the verbalization layer.

    The text field is always the validated outward expression.
    """

    text: str
    model_name: str
    used_llm: bool
    validation_status: str
    validation_reasons: tuple[str, ...]
    candidate_text: str
    canonical_text: str
    attempt_count: int
    first_candidate_text: str
    retry_candidate_text: str | None


class LLMBackend(Protocol):
    """
    Minimal protocol for any language backend.
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

    Any generative candidate is subordinate to:
        ExpressionDecision
        + canonical expression
        + ExpressionValidator

    A generative backend receives at most one corrective retry.

    The retry may repair linguistic preservation of already admitted
    claims, but it may not create, strengthen, or reinterpret RDA
    evidence.
    """

    def __init__(
        self,
        backend: LLMBackend | None = None,
        validator: ExpressionValidator | None = None,
        *,
        allow_retry: bool = True,
    ) -> None:
        self._backend = backend or CanonicalBackend()
        self._adapter = LanguageAdapter()
        self._validator = validator or ExpressionValidator()
        self._allow_retry = allow_retry

    def speak(
        self,
        decision: ExpressionDecision,
        *,
        language: str = "ru",
    ) -> VerbalizationResult:
        canonical = self._adapter.render(decision)

        first_request = VerbalizationRequest(
            decision=decision,
            canonical_text=canonical.text,
            language=language,
        )

        first_candidate = self._backend.generate(first_request)

        first_validation = self._validator.validate(
            decision=decision,
            canonical_text=canonical.text,
            candidate_text=first_candidate,
        )

        is_llm = not isinstance(
            self._backend,
            CanonicalBackend,
        )

        if (
            first_validation.status is ValidationStatus.ACCEPTED
            or not is_llm
            or not self._allow_retry
            or not canonical.text.strip()
        ):
            return VerbalizationResult(
                text=first_validation.output_text,
                model_name=self._backend.__class__.__name__,
                used_llm=is_llm,
                validation_status=first_validation.status.value,
                validation_reasons=first_validation.reasons,
                candidate_text=first_candidate,
                canonical_text=canonical.text,
                attempt_count=1,
                first_candidate_text=first_candidate,
                retry_candidate_text=None,
            )

        retry_request = VerbalizationRequest(
            decision=decision,
            canonical_text=canonical.text,
            language=language,
            retry_reasons=first_validation.reasons,
            previous_candidate=first_candidate,
        )

        retry_candidate = self._backend.generate(retry_request)

        retry_validation = self._validator.validate(
            decision=decision,
            canonical_text=canonical.text,
            candidate_text=retry_candidate,
        )

        return VerbalizationResult(
            text=retry_validation.output_text,
            model_name=self._backend.__class__.__name__,
            used_llm=True,
            validation_status=retry_validation.status.value,
            validation_reasons=retry_validation.reasons,
            candidate_text=retry_candidate,
            canonical_text=canonical.text,
            attempt_count=2,
            first_candidate_text=first_candidate,
            retry_candidate_text=retry_candidate,
        )
