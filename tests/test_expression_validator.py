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

def test_scientific_notation_preserves_supported_deviation_score():
    from roif.development.expression_gate import (
        ExpressionDecision,
        ExpressionEvidence,
        ExpressionKind,
    )

    validator = ExpressionValidator()

    evidence = ExpressionEvidence(
        sequence_index=2000,
        timestamp=1.9999999999998905,
        deviation_score=999999999.9999944,
        maximum_absolute_deviation=None,
        l2_deviation_norm=None,
        group_count=0,
        strongest_group_strength=None,
        represented_dimension=None,
        residual_variance_fraction=None,
        persistent_residual_dimension_count=None,
        growth_supported=None,
    )

    decision = ExpressionDecision(
        kind=ExpressionKind.STATE_CHANGE,
        evidence=evidence,
    )

    canonical = (
        "Моё текущее внутреннее состояние отличается от ранее "
        "сформированной знакомой регулярности. "
        "Величина внутреннего отклонения: 1e+09."
    )

    result = validator.validate(
        decision=decision,
        canonical_text=canonical,
        candidate_text=canonical,
    )

    assert result.status is ValidationStatus.ACCEPTED
    assert result.reasons == ()
    assert result.output_text == canonical


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


def _difference_canonical_text() -> str:
    return (
        "Обнаруженное внутреннее различие имеет распределённую "
        "структуру по моим внутренним каналам. "
        "Максимальное абсолютное отклонение: 1e+09. "
        "Общая величина многоканального отклонения: 1000000021.0185."
    )


def test_difference_structure_canonical_expression_is_accepted():
    validator = ExpressionValidator()

    canonical = _difference_canonical_text()

    result = validator.validate(
        decision=_difference_decision(),
        canonical_text=canonical,
        candidate_text=canonical,
    )

    assert result.status is ValidationStatus.ACCEPTED
    assert result.reasons == ()
    assert result.output_text == canonical


def test_difference_structure_safe_rephrasing_is_accepted():
    validator = ExpressionValidator()

    candidate = (
        "Внутреннее различие имеет структуру по нескольким каналам. "
        "Максимальное абсолютное отклонение равно 1e+09, "
        "а L2-норма отклонения равна 1000000021.0185."
    )

    result = validator.validate(
        decision=_difference_decision(),
        canonical_text=_difference_canonical_text(),
        candidate_text=candidate,
    )

    assert result.status is ValidationStatus.ACCEPTED
    assert result.reasons == ()
    assert result.output_text == candidate


def test_difference_structure_missing_maximum_deviation_is_rejected():
    validator = ExpressionValidator()

    candidate = (
        "Обнаруженное внутреннее различие имеет структуру. "
        "L2-норма отклонения равна 1000000021.0185."
    )

    result = validator.validate(
        decision=_difference_decision(),
        canonical_text=_difference_canonical_text(),
        candidate_text=candidate,
    )

    assert result.status is ValidationStatus.REJECTED

    assert (
        "missing_required_numeric_claim:"
        "maximum_absolute_deviation"
        in result.reasons
    )


def test_difference_structure_missing_l2_norm_is_rejected():
    validator = ExpressionValidator()

    candidate = (
        "Обнаруженное внутреннее различие имеет структуру. "
        "Максимальное абсолютное отклонение равно 1e+09."
    )

    result = validator.validate(
        decision=_difference_decision(),
        canonical_text=_difference_canonical_text(),
        candidate_text=candidate,
    )

    assert result.status is ValidationStatus.REJECTED

    assert (
        "missing_required_numeric_claim:"
        "l2_deviation_norm"
        in result.reasons
    )


def test_difference_structure_unsupported_number_is_rejected():
    validator = ExpressionValidator()

    candidate = (
        "Обнаруженное внутреннее различие имеет структуру. "
        "Максимальное абсолютное отклонение равно 1e+09, "
        "L2-норма отклонения равна 1000000021.0185, "
        "а дополнительная величина равна 7."
    )

    result = validator.validate(
        decision=_difference_decision(),
        canonical_text=_difference_canonical_text(),
        candidate_text=candidate,
    )

    assert result.status is ValidationStatus.REJECTED
    assert "unsupported_numeric_claim:7.0" in result.reasons
