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
