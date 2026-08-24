from roif.development.expression_gate import (
    ExpressionGate,
    ExpressionKind,
)
from roif.development.familiar_state_change import ChangeDetection
from roif.development.endogenous_difference import (
    ChannelDeviation,
    DifferenceProfile,
)
from roif.development.endogenous_grouping import (
    EndogenousGroup,
    GroupingResult,
)
from roif.development.dimensional_growth import (
    DimensionalGrowthAssessment,
    ResidualDimension,
)


def test_no_expression_without_supported_state():
    gate = ExpressionGate()

    decision = gate.decide()

    assert decision.kind is ExpressionKind.NO_EXPRESSION
    assert decision.should_express is False


def test_state_change_expression():
    gate = ExpressionGate()

    change = ChangeDetection(
        sequence_index=10,
        timestamp=1.5,
        deviation_score=12.0,
        changed=True,
    )

    decision = gate.decide(change=change)

    assert decision.kind is ExpressionKind.STATE_CHANGE
    assert decision.should_express is True
    assert decision.evidence.sequence_index == 10
    assert decision.evidence.timestamp == 1.5
    assert decision.evidence.deviation_score == 12.0


def test_relational_structure_has_priority_over_state_change():
    gate = ExpressionGate()

    change = ChangeDetection(
        sequence_index=11,
        timestamp=2.0,
        deviation_score=15.0,
        changed=True,
    )

    difference = DifferenceProfile(
        timestamp=2.0,
        sequence_index=11,
        channel_deviations=(
            ChannelDeviation(
                channel_name="channel_a",
                current_value=2.0,
                familiar_mean=1.0,
                familiar_scale=0.5,
                signed_deviation=2.0,
                absolute_deviation=2.0,
            ),
            ChannelDeviation(
                channel_name="channel_b",
                current_value=0.0,
                familiar_mean=1.0,
                familiar_scale=0.5,
                signed_deviation=-2.0,
                absolute_deviation=2.0,
            ),
        ),
    )

    grouping = GroupingResult(
        channel_names=("channel_a", "channel_b"),
        relations=(),
        groups=(
            EndogenousGroup(
                group_id=0,
                channel_names=("channel_a", "channel_b"),
                internal_relation_strength=0.97,
            ),
        ),
    )

    decision = gate.decide(
        change=change,
        difference=difference,
        grouping=grouping,
    )

    assert decision.kind is ExpressionKind.RELATIONAL_STRUCTURE
    assert decision.evidence.group_count == 1
    assert decision.evidence.strongest_group_strength == 0.97
    assert decision.evidence.maximum_absolute_deviation == 2.0


def test_representational_insufficiency_has_highest_priority():
    gate = ExpressionGate()

    residual_dimension = ResidualDimension(
        residual_index=0,
        eigenvalue=1.0,
        explained_residual_fraction=1.0,
        cumulative_residual_fraction=1.0,
        direction=(1.0, 0.0, 0.0),
    )

    assessment = DimensionalGrowthAssessment(
        sample_count=100,
        observed_dimension=5,
        represented_dimension=3,
        residual_numerical_rank=2,
        residual_effective_rank=1.5,
        residual_participation_ratio=1.4,
        total_variance=10.0,
        represented_variance=5.4,
        residual_variance=4.6,
        residual_variance_fraction=0.46,
        reconstruction_error_rms=0.2,
        normalized_reconstruction_error=0.3,
        persistent_residual_dimension_count=1,
        growth_supported=True,
        residual_dimensions=(residual_dimension,),
    )

    decision = gate.decide(
        dimensional_growth=assessment,
    )

    assert (
        decision.kind
        is ExpressionKind.REPRESENTATIONAL_INSUFFICIENCY
    )
    assert decision.should_express is True
    assert decision.evidence.represented_dimension == 3
    assert decision.evidence.residual_variance_fraction == 0.46
    assert (
        decision.evidence.persistent_residual_dimension_count
        == 1
    )
    assert decision.evidence.growth_supported is True