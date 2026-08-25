from roif.development.expression_gate import (
    ExpressionDecision,
    ExpressionEvidence,
    ExpressionKind,
)
from roif.development.language_adapter import (
    LanguageAdapter,
)


def make_evidence(**overrides):
    values = {
        "sequence_index": None,
        "timestamp": None,
        "deviation_score": None,
        "maximum_absolute_deviation": None,
        "l2_deviation_norm": None,
        "group_count": 0,
        "strongest_group_strength": None,
        "represented_dimension": None,
        "residual_variance_fraction": None,
        "persistent_residual_dimension_count": None,
        "growth_supported": None,
    }
    values.update(overrides)
    return ExpressionEvidence(**values)


def test_no_expression_renders_empty_text():
    adapter = LanguageAdapter()

    decision = ExpressionDecision(
        kind=ExpressionKind.NO_EXPRESSION,
        evidence=make_evidence(),
    )

    result = adapter.render(decision)

    assert result.text == ""
    assert result.kind is ExpressionKind.NO_EXPRESSION


def test_state_change_renders_traceable_text():
    adapter = LanguageAdapter()

    decision = ExpressionDecision(
        kind=ExpressionKind.STATE_CHANGE,
        evidence=make_evidence(
            sequence_index=10,
            timestamp=1.5,
            deviation_score=12.0,
        ),
    )

    result = adapter.render(decision)

    assert "отличается" in result.text
    assert result.trace["deviation_score"] == 12.0


def test_difference_structure_renders_profile_information():
    adapter = LanguageAdapter()

    decision = ExpressionDecision(
        kind=ExpressionKind.DIFFERENCE_STRUCTURE,
        evidence=make_evidence(
            sequence_index=2000,
            timestamp=1.9999999999998905,
            maximum_absolute_deviation=999999999.9999944,
            l2_deviation_norm=1000000021.0185108,
        ),
    )

    result = adapter.render(decision)

    assert result.kind is ExpressionKind.DIFFERENCE_STRUCTURE
    assert "1e+09" in result.text
    assert "1000000021.0185" in result.text
    assert (
        result.trace["maximum_absolute_deviation"]
        == 999999999.9999944
    )
    assert (
        result.trace["l2_deviation_norm"]
        == 1000000021.0185108
    )

def test_relational_structure_renders_group_information():
    adapter = LanguageAdapter()

    decision = ExpressionDecision(
        kind=ExpressionKind.RELATIONAL_STRUCTURE,
        evidence=make_evidence(
            group_count=2,
            strongest_group_strength=0.97,
        ),
    )

    result = adapter.render(decision)

    assert "совместная структура" in result.text
    assert result.trace["group_count"] == 2
    assert result.trace["strongest_group_strength"] == 0.97


def test_representational_insufficiency_renders_only_evidence():
    adapter = LanguageAdapter()

    decision = ExpressionDecision(
        kind=ExpressionKind.REPRESENTATIONAL_INSUFFICIENCY,
        evidence=make_evidence(
            represented_dimension=3,
            residual_variance_fraction=0.4576,
            persistent_residual_dimension_count=2,
            growth_supported=True,
        ),
    )

    result = adapter.render(decision)

    assert "недостаточно" in result.text
    assert "3" in result.text
    assert "2" in result.text
    assert result.trace["growth_supported"] is True