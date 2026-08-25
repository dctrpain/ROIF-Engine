from __future__ import annotations

from roif.development.expression_gate import (
    ExpressionDecision,
    ExpressionEvidence,
    ExpressionKind,
)
from roif.development.language_adapter import LanguageAdapter
from roif.development.llm_verbalizer import (
    CanonicalBackend,
    LLMVerbalizer,
    VerbalizationRequest,
)


def _difference_decision() -> ExpressionDecision:
    evidence = ExpressionEvidence(
        sequence_index=2000,
        timestamp=1.9999999999998905,
        deviation_score=None,
        maximum_absolute_deviation=999999999.9999944,
        l2_deviation_norm=1000000021.0185108,
        group_count=0,
        strongest_group_strength=None,
        represented_dimension=None,
        residual_variance_fraction=None,
        persistent_residual_dimension_count=None,
        growth_supported=None,
    )

    return ExpressionDecision(
        kind=ExpressionKind.DIFFERENCE_STRUCTURE,
        evidence=evidence,
    )

def _growth_decision() -> ExpressionDecision:
    evidence = ExpressionEvidence(
        sequence_index=None,
        timestamp=None,
        deviation_score=None,
        maximum_absolute_deviation=None,
        l2_deviation_norm=None,
        group_count=0,
        strongest_group_strength=None,
        represented_dimension=3,
        residual_variance_fraction=0.45759965414789505,
        persistent_residual_dimension_count=2,
        growth_supported=True,
    )

    return ExpressionDecision(
        kind=ExpressionKind.REPRESENTATIONAL_INSUFFICIENCY,
        evidence=evidence,
    )


def test_canonical_backend_returns_canonical_text_unchanged():
    decision = _growth_decision()

    adapter = LanguageAdapter()
    canonical = adapter.render(decision)

    request = VerbalizationRequest(
        decision=decision,
        canonical_text=canonical.text,
        language="ru",
    )

    backend = CanonicalBackend()

    assert backend.generate(request) == canonical.text


def test_default_verbalizer_uses_canonical_backend():
    decision = _growth_decision()

    adapter = LanguageAdapter()
    canonical = adapter.render(decision)

    verbalizer = LLMVerbalizer()
    result = verbalizer.speak(decision)

    assert result.text == canonical.text
    assert result.model_name == "CanonicalBackend"
    assert result.used_llm is False


def test_verbalizer_preserves_traceable_growth_content():
    decision = _growth_decision()

    verbalizer = LLMVerbalizer()
    result = verbalizer.speak(decision)

    assert "3" in result.text
    assert "2" in result.text
    assert "0.4576" in result.text


def test_backend_receives_original_expression_decision():
    decision = _growth_decision()

    class RecordingBackend:
        def __init__(self) -> None:
            self.request = None

        def generate(
            self,
            request: VerbalizationRequest,
        ) -> str:
            self.request = request
            return request.canonical_text

    backend = RecordingBackend()
    verbalizer = LLMVerbalizer(backend=backend)

    result = verbalizer.speak(decision)

    assert backend.request is not None
    assert backend.request.decision is decision
    assert backend.request.language == "ru"
    assert backend.request.canonical_text == result.text

    assert result.model_name == "RecordingBackend"
    assert result.used_llm is True

def test_retry_is_not_used_when_first_candidate_is_accepted():
    decision = _growth_decision()

    class AcceptedBackend:
        def __init__(self) -> None:
            self.calls = 0

        def generate(
            self,
            request: VerbalizationRequest,
        ) -> str:
            self.calls += 1
            return request.canonical_text

    backend = AcceptedBackend()
    verbalizer = LLMVerbalizer(backend=backend)

    result = verbalizer.speak(decision)

    assert backend.calls == 1
    assert result.attempt_count == 1
    assert result.validation_status == "accepted"
    assert result.retry_candidate_text is None
    assert result.text == result.canonical_text


def test_retry_is_used_once_and_second_candidate_can_be_accepted():
    decision = _growth_decision()

    safe_retry = (
        "Моего текущего представления недостаточно, чтобы полностью "
        "описать устойчиво повторяющуюся структуру моего опыта. "
        "Сейчас представлена размерность 3, сохраняются "
        "2 дополнительных остаточных направления, а доля "
        "необъяснённой вариативности равна 0.4576."
    )

    class RetryBackend:
        def __init__(self) -> None:
            self.requests = []

        def generate(
            self,
            request: VerbalizationRequest,
        ) -> str:
            self.requests.append(request)

            if len(self.requests) == 1:
                return (
                    "Моё внутреннее представление недостаточно "
                    "для полного описания моего опыта. "
                    "Сейчас представлена размерность 3, "
                    "с двумя дополнительными остаточными направлениями "
                    "и 0.4576 необъяснённой вариативности."
                )

            return safe_retry

    backend = RetryBackend()
    verbalizer = LLMVerbalizer(backend=backend)

    result = verbalizer.speak(decision)

    assert len(backend.requests) == 2

    first_request = backend.requests[0]
    retry_request = backend.requests[1]

    assert first_request.is_retry is False
    assert retry_request.is_retry is True

    assert (
        "missing_required_claim:"
        "repeating_experience_structure_present"
        in retry_request.retry_reasons
    )

    assert retry_request.previous_candidate is not None

    assert result.attempt_count == 2
    assert result.validation_status == "accepted"
    assert result.text == safe_retry
    assert result.first_candidate_text != safe_retry
    assert result.retry_candidate_text == safe_retry


def test_retry_is_limited_to_one_and_falls_back_to_canonical():
    decision = _growth_decision()

    class AlwaysInvalidBackend:
        def __init__(self) -> None:
            self.calls = 0

        def generate(
            self,
            request: VerbalizationRequest,
        ) -> str:
            self.calls += 1

            return (
                "Я чувствую, что моего представления недостаточно. "
                "Сейчас представлена размерность 7."
            )

    backend = AlwaysInvalidBackend()
    verbalizer = LLMVerbalizer(backend=backend)

    result = verbalizer.speak(decision)

    assert backend.calls == 2
    assert result.attempt_count == 2
    assert result.validation_status == "rejected"

    assert result.text == result.canonical_text

    assert result.retry_candidate_text is not None

    assert any(
        reason.startswith("unsupported_semantic_term:")
        for reason in result.validation_reasons
    )

    assert any(
        reason.startswith("unsupported_numeric_claim:")
        for reason in result.validation_reasons
    )


def test_verbalizer_accepts_traceable_difference_structure():
    decision = _difference_decision()

    safe_candidate = (
        "Внутреннее различие имеет структуру. "
        "Максимальное абсолютное отклонение равно 1e+09, "
        "а L2-норма отклонения равна 1000000021.0185."
    )

    class DifferenceBackend:
        def generate(
            self,
            request: VerbalizationRequest,
        ) -> str:
            return safe_candidate

    verbalizer = LLMVerbalizer(
        backend=DifferenceBackend(),
    )

    result = verbalizer.speak(decision)

    assert result.validation_status == "accepted"
    assert result.attempt_count == 1
    assert result.text == safe_candidate
    assert "1e+09" in result.text
    assert "1000000021.0185" in result.text


def test_difference_structure_retry_restores_missing_claim():
    decision = _difference_decision()

    safe_retry = (
        "Внутреннее различие имеет структуру. "
        "Максимальное абсолютное отклонение равно 1e+09, "
        "а L2-норма отклонения равна 1000000021.0185."
    )

    class RetryDifferenceBackend:
        def __init__(self) -> None:
            self.requests = []

        def generate(
            self,
            request: VerbalizationRequest,
        ) -> str:
            self.requests.append(request)

            if len(self.requests) == 1:
                return (
                    "Внутреннее различие имеет структуру. "
                    "L2-норма отклонения равна "
                    "1000000021.0185."
                )

            return safe_retry

    backend = RetryDifferenceBackend()
    verbalizer = LLMVerbalizer(backend=backend)

    result = verbalizer.speak(decision)

    assert len(backend.requests) == 2
    assert backend.requests[0].is_retry is False
    assert backend.requests[1].is_retry is True
    assert (
        "missing_required_numeric_claim:"
        "maximum_absolute_deviation"
        in backend.requests[1].retry_reasons
    )
    assert result.attempt_count == 2
    assert result.validation_status == "accepted"
    assert result.text == safe_retry
