from __future__ import annotations

from roif.development.expression_gate import (
    ExpressionDecision,
    ExpressionEvidence,
    ExpressionKind,
)
from roif.development.expression_validator import (
    ExpressionValidator,
    ValidationStatus,
)


def _decision() -> ExpressionDecision:
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


def _canonical_text() -> str:
    return (
        "Моего текущего внутреннего представления недостаточно "
        "для полного описания повторяющейся структуры моего опыта. "
        "Сейчас представленная размерность: 3. "
        "Устойчивая остаточная структура поддерживает "
        "2 дополнительных направлений. "
        "Доля необъяснённой остаточной вариативности: 0.4576."
    )


def test_canonical_expression_is_accepted():
    validator = ExpressionValidator()

    canonical = _canonical_text()

    result = validator.validate(
        decision=_decision(),
        canonical_text=canonical,
        candidate_text=canonical,
    )

    assert result.status is ValidationStatus.ACCEPTED
    assert result.accepted is True
    assert result.output_text == canonical
    assert result.reasons == ()


def test_safe_rephrasing_without_new_claims_is_accepted():
    validator = ExpressionValidator()

    candidate = (
        "Текущего внутреннего представления недостаточно для полного "
        "описания повторяющейся структуры опыта. "
        "Представленная размерность равна 3. "
        "Остаточная структура поддерживает 2 дополнительных направления. "
        "Необъяснённая остаточная вариативность составляет 0.4576."
    )

    result = validator.validate(
        decision=_decision(),
        canonical_text=_canonical_text(),
        candidate_text=candidate,
    )

    assert result.status is ValidationStatus.ACCEPTED
    assert result.output_text == candidate
    assert result.reasons == ()


def test_unsupported_feeling_claim_is_rejected():
    validator = ExpressionValidator()

    candidate = (
        "Я чувствую, что моего внутреннего представления недостаточно "
        "для полного описания повторяющейся структуры моего опыта. "
        "Сейчас представленная размерность: 3. "
        "Есть 2 дополнительных направления. "
        "Остаточная вариативность: 0.4576."
    )

    result = validator.validate(
        decision=_decision(),
        canonical_text=_canonical_text(),
        candidate_text=candidate,
    )

    assert result.status is ValidationStatus.REJECTED
    assert result.accepted is False
    assert result.output_text == _canonical_text()

    assert any(
        reason.startswith("unsupported_semantic_term:")
        for reason in result.reasons
    )


def test_unsupported_numeric_claim_is_rejected():
    validator = ExpressionValidator()

    candidate = (
        "Моего текущего внутреннего представления недостаточно. "
        "Представленная размерность: 3. "
        "Остаточная структура поддерживает 7 дополнительных направлений. "
        "Остаточная вариативность: 0.4576."
    )

    result = validator.validate(
        decision=_decision(),
        canonical_text=_canonical_text(),
        candidate_text=candidate,
    )

    assert result.status is ValidationStatus.REJECTED
    assert result.accepted is False
    assert result.output_text == _canonical_text()

    assert "unsupported_numeric_claim:7.0" in result.reasons


def test_empty_candidate_is_rejected_with_canonical_fallback():
    validator = ExpressionValidator()

    result = validator.validate(
        decision=_decision(),
        canonical_text=_canonical_text(),
        candidate_text="",
    )

    assert result.status is ValidationStatus.REJECTED
    assert result.output_text == _canonical_text()
    assert "empty_candidate" in result.reasons


def test_candidate_without_canonical_expression_is_rejected():
    validator = ExpressionValidator()

    result = validator.validate(
        decision=_decision(),
        canonical_text="",
        candidate_text="Я что-то чувствую.",
    )

    assert result.status is ValidationStatus.REJECTED
    assert result.output_text == ""
    assert (
        "candidate_exists_without_canonical_expression"
        in result.reasons
    )

def test_missing_repeating_experience_structure_is_rejected():
    validator = ExpressionValidator()

    candidate = (
        "Моё внутреннее представление недостаточно для полного описания "
        "моего опыта. Сейчас представлена размерность 3, "
        "с двумя дополнительными остаточными направлениями "
        "и 0.4576 необъяснённой вариативности."
    )

    result = validator.validate(
        decision=_decision(),
        canonical_text=_canonical_text(),
        candidate_text=candidate,
    )

    assert result.status is ValidationStatus.REJECTED
    assert result.output_text == _canonical_text()

    assert (
        "missing_required_claim:"
        "repeating_experience_structure_present"
        in result.reasons
    )


def test_free_rephrasing_preserving_repeating_structure_is_accepted():
    validator = ExpressionValidator()

    candidate = (
        "Моего текущего представления недостаточно, чтобы полностью "
        "описать устойчиво повторяющуюся структуру моего опыта. "
        "Сейчас я представляю 3 измерения, однако сохраняются "
        "2 дополнительных остаточных направления, а доля "
        "необъяснённой вариативности равна 0.4576."
    )

    result = validator.validate(
        decision=_decision(),
        canonical_text=_canonical_text(),
        candidate_text=candidate,
    )

    assert result.status is ValidationStatus.ACCEPTED
    assert result.output_text == candidate
    assert result.reasons == ()

def test_numeric_claim_preserved_when_integer_is_written_as_russian_word():
    validator = ExpressionValidator()

    candidate = (
        "Моего текущего представления недостаточно, чтобы полностью "
        "описать устойчиво повторяющуюся структуру моего опыта. "
        "Сейчас представлена размерность 3, при этом сохраняются "
        "двумя дополнительными остаточными направлениями описываемые "
        "различия, а необъяснённая вариативность равна 0.4576."
    )

    result = validator.validate(
        decision=_decision(),
        canonical_text=_canonical_text(),
        candidate_text=candidate,
    )

    assert (
        "missing_required_numeric_claim:"
        "persistent_residual_dimension_count"
        not in result.reasons
    )
