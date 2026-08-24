from experiments.roif_rda_3_first_dimensional_growth import (
    run_first_dimensional_growth,
)
from roif.development.expression_gate import (
    ExpressionGate,
    ExpressionKind,
)
from roif.development.language_adapter import LanguageAdapter


EXPECTED_RUN_SHA256 = (
    "3423ebb7346656f6336aed9a68d05bbbf5913763c0612f5c91dee656b47ab24a"
)


def test_real_rda3_first_expression_is_traceable():
    result, assessment = run_first_dimensional_growth(
        return_internal=True,
    )

    assert result["run_sha256"] == EXPECTED_RUN_SHA256

    assert assessment.growth_supported is True
    assert assessment.represented_dimension == 3
    assert assessment.persistent_residual_dimension_count == 2

    gate = ExpressionGate()

    decision = gate.decide(
        dimensional_growth=assessment,
    )

    assert (
        decision.kind
        is ExpressionKind.REPRESENTATIONAL_INSUFFICIENCY
    )

    assert decision.evidence.represented_dimension == 3
    assert (
        decision.evidence.persistent_residual_dimension_count
        == 2
    )

    adapter = LanguageAdapter()

    expression = adapter.render(decision)

    assert expression.text
    assert (
        expression.trace["kind"]
        == "representational_insufficiency"
    )
    assert expression.trace["growth_supported"] is True

    assert (
        expression.trace["residual_variance_fraction"]
        == assessment.residual_variance_fraction
    )
