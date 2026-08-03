from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from roif.history.history_decoder import (
    HistoryHypothesis,
    HypothesisStatus,
)
from roif.history.hypothesis_generator import (
    CandidateStatus,
    CausalRule,
    GeneratedCandidate,
    HypothesisGenerationResult,
    HypothesisGenerator,
    HypothesisGeneratorError,
    HypothesisSeed,
    RuleMatchMode,
)
from roif.history.structural_signature import (
    StructuralSignature,
)


def make_signature(
    signature_id: str = "observed-signature",
    **overrides,
) -> StructuralSignature:
    data = {
        "signature_id": signature_id,
        "source_pattern_id": f"pattern-{signature_id}",
        "label": signature_id,
        "plane_profile": {
            "mechanical": 0.70,
            "thermal": 0.30,
        },
        "agent_profile": {
            "steel_object": 0.60,
            "environment": 0.40,
        },
        "kind_profile": {
            "damage": 0.75,
            "remodeling": 0.25,
        },
        "time_scale_profile": {
            "medium": 0.70,
            "slow": 0.30,
        },
        "target_profile": {
            "element:A": 0.50,
            "element:B": 0.30,
            "element:C": 0.20,
        },
        "total_changes": 3,
        "root_count": 1,
        "leaf_count": 1,
        "causal_depth": 3,
        "start_time": 1.0,
        "end_time": 4.0,
        "duration": 3.0,
        "total_capacity_loss": 0.30,
        "total_capacity_gain": 0.05,
        "net_capacity_effect": -0.25,
        "cumulative_signature_weight": 1.80,
        "mean_signature_weight": 0.60,
        "persistence_index": 0.75,
        "irreversibility_index": 0.65,
        "progression_index": 0.80,
        "adaptation_index": 0.20,
        "harmful_fraction": 0.75,
        "beneficial_fraction": 0.25,
        "mixed_fraction": 0.0,
        "metadata": {
            "source": "test",
        },
    }

    data.update(overrides)
    return StructuralSignature(**data)


def make_rule(
    rule_id: str = "rule-mechanical-impact",
    **overrides,
) -> CausalRule:
    data = {
        "rule_id": rule_id,
        "name": "Mechanical impact rule",
        "hypothesis_name": "Single mechanical impact",
        "description": (
            "Candidate history initiated by mechanical impact."
        ),
        "initiating_plane_ids": (
            "mechanical",
        ),
        "initiating_agent_ids": (
            "steel_object",
        ),
        "candidate_root_ids": (
            "element:A",
        ),
        "plane_thresholds": {
            "mechanical": 0.20,
        },
        "agent_thresholds": {
            "steel_object": 0.10,
        },
        "kind_thresholds": {
            "damage": 0.10,
        },
        "time_scale_thresholds": {
            "medium": 0.10,
        },
        "target_thresholds": {
            "element:A": 0.10,
        },
        "match_mode": RuleMatchMode.ALL,
        "minimum_total_changes": 1,
        "minimum_causal_depth": 1,
        "minimum_capacity_loss": 0.05,
        "minimum_capacity_gain": 0.0,
        "minimum_persistence": 0.20,
        "minimum_irreversibility": 0.20,
        "prior_probability": 0.60,
        "validation_score": 0.90,
        "complexity": 1.0,
        "custom_penalty": 0.05,
        "enabled": True,
        "metadata": {
            "domain": "mechanical",
        },
    }

    data.update(overrides)
    return CausalRule(**data)


def make_seed(
    seed_id: str = "seed-001",
    **overrides,
) -> HypothesisSeed:
    data = {
        "seed_id": seed_id,
        "rule_id": "rule-mechanical-impact",
        "name": "Single mechanical impact",
        "description": "Prediction seed.",
        "initiating_plane_ids": (
            "mechanical",
        ),
        "initiating_agent_ids": (
            "steel_object",
        ),
        "candidate_root_ids": (
            "element:A",
        ),
        "prior_probability": 0.60,
        "validation_score": 0.90,
        "complexity": 1.0,
        "custom_penalty": 0.05,
        "metadata": {
            "source": "test",
        },
    }

    data.update(overrides)
    return HypothesisSeed(**data)


def make_hypothesis(
    hypothesis_id: str = "seed-001",
) -> HistoryHypothesis:
    return HistoryHypothesis(
        hypothesis_id=hypothesis_id,
        name="Single mechanical impact",
        description="Predicted candidate.",
        predicted_signature=make_signature(
            signature_id=f"predicted-{hypothesis_id}"
        ),
        initiating_plane_ids=("mechanical",),
        initiating_agent_ids=("steel_object",),
        candidate_root_ids=("element:A",),
        prior_probability=0.60,
        validation_score=0.90,
        complexity=1.0,
        custom_penalty=0.05,
        status=HypothesisStatus.FORWARD_VALIDATED,
    )


def test_causal_rule_creation() -> None:
    rule = make_rule()

    assert rule.rule_id == "rule-mechanical-impact"
    assert rule.name == "Mechanical impact rule"
    assert rule.hypothesis_name == (
        "Single mechanical impact"
    )
    assert rule.match_mode is RuleMatchMode.ALL
    assert rule.minimum_total_changes == 1
    assert rule.minimum_causal_depth == 1
    assert rule.prior_probability == pytest.approx(0.60)
    assert rule.validation_score == pytest.approx(0.90)
    assert rule.enabled is True


def test_causal_rule_accepts_string_match_mode() -> None:
    rule = make_rule(
        match_mode="any",
    )

    assert rule.match_mode is RuleMatchMode.ANY


def test_causal_rule_is_immutable() -> None:
    rule = make_rule()

    with pytest.raises(FrozenInstanceError):
        rule.enabled = False


def test_causal_rule_metadata_is_read_only() -> None:
    rule = make_rule()

    with pytest.raises(TypeError):
        rule.metadata["new"] = "value"


def test_causal_rule_thresholds_are_read_only() -> None:
    rule = make_rule()

    with pytest.raises(TypeError):
        rule.plane_thresholds["thermal"] = 0.2


def test_causal_rule_ids_are_deduplicated() -> None:
    rule = make_rule(
        initiating_plane_ids=(
            "mechanical",
            "thermal",
            "mechanical",
        ),
        initiating_agent_ids=(
            "steel",
            "wood",
            "steel",
        ),
        candidate_root_ids=(
            "A",
            "B",
            "A",
        ),
    )

    assert rule.initiating_plane_ids == (
        "mechanical",
        "thermal",
    )
    assert rule.initiating_agent_ids == (
        "steel",
        "wood",
    )
    assert rule.candidate_root_ids == (
        "A",
        "B",
    )


def test_rule_has_profile_constraints() -> None:
    rule = make_rule()

    assert rule.has_profile_constraints is True


def test_rule_without_profile_constraints() -> None:
    rule = make_rule(
        plane_thresholds={},
        agent_thresholds={},
        kind_thresholds={},
        time_scale_thresholds={},
        target_thresholds={},
    )

    assert rule.has_profile_constraints is False


def test_rule_matches_observed_signature() -> None:
    rule = make_rule()
    observed = make_signature()

    assert rule.matches(observed) is True


def test_disabled_rule_does_not_match() -> None:
    rule = make_rule(
        enabled=False,
    )

    assert rule.matches(make_signature()) is False


def test_rule_rejects_insufficient_total_changes() -> None:
    rule = make_rule(
        minimum_total_changes=4,
    )

    assert rule.matches(make_signature()) is False


def test_rule_rejects_insufficient_causal_depth() -> None:
    rule = make_rule(
        minimum_causal_depth=4,
    )

    assert rule.matches(make_signature()) is False


def test_rule_rejects_insufficient_capacity_loss() -> None:
    rule = make_rule(
        minimum_capacity_loss=0.50,
    )

    assert rule.matches(make_signature()) is False


def test_rule_rejects_insufficient_capacity_gain() -> None:
    rule = make_rule(
        minimum_capacity_gain=0.20,
    )

    assert rule.matches(make_signature()) is False


def test_rule_rejects_insufficient_persistence() -> None:
    rule = make_rule(
        minimum_persistence=0.90,
    )

    assert rule.matches(make_signature()) is False


def test_rule_rejects_insufficient_irreversibility() -> None:
    rule = make_rule(
        minimum_irreversibility=0.90,
    )

    assert rule.matches(make_signature()) is False


def test_rule_all_mode_requires_all_profile_checks() -> None:
    rule = make_rule(
        plane_thresholds={
            "mechanical": 0.20,
        },
        agent_thresholds={
            "missing_agent": 0.10,
        },
        kind_thresholds={},
        time_scale_thresholds={},
        target_thresholds={},
        match_mode=RuleMatchMode.ALL,
    )

    assert rule.matches(make_signature()) is False


def test_rule_any_mode_requires_one_profile_check() -> None:
    rule = make_rule(
        plane_thresholds={
            "mechanical": 0.20,
        },
        agent_thresholds={
            "missing_agent": 0.10,
        },
        kind_thresholds={},
        time_scale_thresholds={},
        target_thresholds={},
        match_mode=RuleMatchMode.ANY,
    )

    assert rule.matches(make_signature()) is True


def test_rule_without_checks_matches_by_global_constraints() -> None:
    rule = make_rule(
        plane_thresholds={},
        agent_thresholds={},
        kind_thresholds={},
        time_scale_thresholds={},
        target_thresholds={},
    )

    assert rule.matches(make_signature()) is True


def test_active_plane_constraint_matches_intersection() -> None:
    rule = make_rule(
        initiating_plane_ids=(
            "mechanical",
        ),
    )

    assert rule.matches(
        make_signature(),
        active_plane_ids=("mechanical", "thermal"),
    ) is True


def test_active_plane_constraint_rejects_no_intersection() -> None:
    rule = make_rule(
        initiating_plane_ids=(
            "mechanical",
        ),
    )

    assert rule.matches(
        make_signature(),
        active_plane_ids=("biological",),
    ) is False


def test_known_agent_constraint() -> None:
    rule = make_rule(
        initiating_agent_ids=(
            "steel_object",
        ),
    )

    assert rule.matches(
        make_signature(),
        known_agent_ids=("steel_object",),
    ) is True

    assert rule.matches(
        make_signature(),
        known_agent_ids=("wood_object",),
    ) is False


def test_allowed_root_constraint() -> None:
    rule = make_rule(
        candidate_root_ids=(
            "element:A",
        ),
    )

    assert rule.matches(
        make_signature(),
        allowed_root_ids=("element:A",),
    ) is True

    assert rule.matches(
        make_signature(),
        allowed_root_ids=("element:Z",),
    ) is False


@pytest.mark.parametrize(
    "field_name",
    (
        "minimum_total_changes",
        "minimum_causal_depth",
    ),
)
def test_rule_rejects_negative_integer_constraints(
    field_name: str,
) -> None:
    with pytest.raises(
        HypothesisGeneratorError,
        match=field_name,
    ):
        make_rule(
            **{
                field_name: -1,
            }
        )


def test_rule_rejects_noninteger_constraints() -> None:
    with pytest.raises(
        HypothesisGeneratorError,
        match="minimum_total_changes must be an integer",
    ):
        make_rule(
            minimum_total_changes=1.5,
        )


@pytest.mark.parametrize(
    "field_name",
    (
        "minimum_capacity_loss",
        "minimum_capacity_gain",
        "complexity",
    ),
)
def test_rule_rejects_negative_nonnegative_fields(
    field_name: str,
) -> None:
    with pytest.raises(
        HypothesisGeneratorError,
        match=field_name,
    ):
        make_rule(
            **{
                field_name: -0.01,
            }
        )


@pytest.mark.parametrize(
    "field_name",
    (
        "minimum_persistence",
        "minimum_irreversibility",
        "prior_probability",
        "validation_score",
        "custom_penalty",
    ),
)
def test_rule_rejects_invalid_unit_interval_fields(
    field_name: str,
) -> None:
    with pytest.raises(
        HypothesisGeneratorError,
        match=field_name,
    ):
        make_rule(
            **{
                field_name: 1.01,
            }
        )


def test_rule_rejects_invalid_threshold() -> None:
    with pytest.raises(
        HypothesisGeneratorError,
        match="plane_thresholds",
    ):
        make_rule(
            plane_thresholds={
                "mechanical": 1.1,
            }
        )


def test_rule_rejects_invalid_match_mode() -> None:
    with pytest.raises(
        HypothesisGeneratorError,
        match="unsupported match mode",
    ):
        make_rule(
            match_mode="unsupported",
        )


def test_rule_rejects_nonboolean_enabled() -> None:
    with pytest.raises(
        HypothesisGeneratorError,
        match="enabled must be a boolean",
    ):
        make_rule(
            enabled=1,
        )


def test_rule_round_trip() -> None:
    rule = make_rule()

    restored = CausalRule.from_dict(
        rule.to_dict()
    )

    assert restored == rule
    assert restored is not rule


def test_rule_serialized_version() -> None:
    data = make_rule().to_dict()

    assert data["version"] == "1.0.0"
    assert data["rule_id"] == "rule-mechanical-impact"
    assert data["match_mode"] == "all"


def test_rule_from_dict_rejects_version() -> None:
    data = make_rule().to_dict()
    data["version"] = "99.0"

    with pytest.raises(
        HypothesisGeneratorError,
        match="unsupported causal rule version",
    ):
        CausalRule.from_dict(data)


def test_seed_creation() -> None:
    seed = make_seed()

    assert seed.seed_id == "seed-001"
    assert seed.rule_id == "rule-mechanical-impact"
    assert seed.name == "Single mechanical impact"
    assert seed.prior_probability == pytest.approx(0.60)
    assert seed.validation_score == pytest.approx(0.90)


def test_seed_is_immutable() -> None:
    seed = make_seed()

    with pytest.raises(FrozenInstanceError):
        seed.name = "Changed"


def test_seed_metadata_is_read_only() -> None:
    seed = make_seed()

    with pytest.raises(TypeError):
        seed.metadata["new"] = "value"


def test_seed_ids_are_deduplicated() -> None:
    seed = make_seed(
        initiating_plane_ids=(
            "mechanical",
            "thermal",
            "mechanical",
        ),
        initiating_agent_ids=(
            "steel",
            "wood",
            "steel",
        ),
        candidate_root_ids=(
            "A",
            "B",
            "A",
        ),
    )

    assert seed.initiating_plane_ids == (
        "mechanical",
        "thermal",
    )
    assert seed.initiating_agent_ids == (
        "steel",
        "wood",
    )
    assert seed.candidate_root_ids == (
        "A",
        "B",
    )


def test_seed_to_dict() -> None:
    data = make_seed().to_dict()

    assert data["version"] == "1.0.0"
    assert data["seed_id"] == "seed-001"
    assert data["rule_id"] == "rule-mechanical-impact"


def test_generated_predicted_candidate() -> None:
    seed = make_seed()
    hypothesis = make_hypothesis()

    candidate = GeneratedCandidate(
        seed=seed,
        status=CandidateStatus.PREDICTED,
        hypothesis=hypothesis,
    )

    assert candidate.status is CandidateStatus.PREDICTED
    assert candidate.hypothesis == hypothesis
    assert candidate.error_message is None


def test_generated_failed_candidate() -> None:
    candidate = GeneratedCandidate(
        seed=make_seed(),
        status=CandidateStatus.FAILED,
        error_message="RuntimeError: failed",
    )

    assert candidate.status is CandidateStatus.FAILED
    assert candidate.hypothesis is None
    assert candidate.error_message == "RuntimeError: failed"


def test_generated_candidate_accepts_string_status() -> None:
    candidate = GeneratedCandidate(
        seed=make_seed(),
        status="skipped",
    )

    assert candidate.status is CandidateStatus.SKIPPED


def test_predicted_candidate_requires_hypothesis() -> None:
    with pytest.raises(
        HypothesisGeneratorError,
        match="predicted candidates require",
    ):
        GeneratedCandidate(
            seed=make_seed(),
            status=CandidateStatus.PREDICTED,
        )


def test_failed_candidate_requires_error_message() -> None:
    with pytest.raises(
        HypothesisGeneratorError,
        match="failed candidates require",
    ):
        GeneratedCandidate(
            seed=make_seed(),
            status=CandidateStatus.FAILED,
        )


def test_generated_candidate_rejects_invalid_seed() -> None:
    with pytest.raises(
        HypothesisGeneratorError,
        match="seed must be a HypothesisSeed",
    ):
        GeneratedCandidate(
            seed="seed",
            status=CandidateStatus.SKIPPED,
        )


def test_generated_candidate_rejects_invalid_hypothesis() -> None:
    with pytest.raises(
        HypothesisGeneratorError,
        match="hypothesis must be a HistoryHypothesis",
    ):
        GeneratedCandidate(
            seed=make_seed(),
            status=CandidateStatus.GENERATED,
            hypothesis="hypothesis",
        )


def test_generation_result_creation() -> None:
    predicted = GeneratedCandidate(
        seed=make_seed("seed-a"),
        status=CandidateStatus.PREDICTED,
        hypothesis=make_hypothesis("seed-a"),
    )
    failed = GeneratedCandidate(
        seed=make_seed("seed-b"),
        status=CandidateStatus.FAILED,
        error_message="ValueError: failure",
    )

    result = HypothesisGenerationResult(
        result_id="generation-001",
        observed_signature_id="observed-signature",
        candidates=(predicted, failed),
        metadata={
            "experiment": "test",
        },
    )

    assert result.result_id == "generation-001"
    assert result.predicted_count == 1
    assert result.failed_count == 1
    assert result.hypotheses == (
        predicted.hypothesis,
    )
    assert result.failed == (failed,)


def test_generation_result_is_immutable() -> None:
    result = HypothesisGenerationResult(
        observed_signature_id="observed",
    )

    with pytest.raises(FrozenInstanceError):
        result.result_id = "changed"


def test_generation_result_metadata_is_read_only() -> None:
    result = HypothesisGenerationResult(
        observed_signature_id="observed",
    )

    with pytest.raises(TypeError):
        result.metadata["new"] = "value"


def test_generation_result_rejects_duplicate_seed_ids() -> None:
    first = GeneratedCandidate(
        seed=make_seed("duplicate"),
        status=CandidateStatus.SKIPPED,
    )
    second = GeneratedCandidate(
        seed=make_seed("duplicate"),
        status=CandidateStatus.SKIPPED,
    )

    with pytest.raises(
        HypothesisGeneratorError,
        match="seed_id values must be unique",
    ):
        HypothesisGenerationResult(
            observed_signature_id="observed",
            candidates=(first, second),
        )


def test_generation_result_to_dict() -> None:
    candidate = GeneratedCandidate(
        seed=make_seed(),
        status=CandidateStatus.PREDICTED,
        hypothesis=make_hypothesis(),
    )

    result = HypothesisGenerationResult(
        result_id="generation-001",
        observed_signature_id="observed",
        candidates=(candidate,),
    )

    data = result.to_dict()

    assert data["version"] == "1.0.0"
    assert data["result_id"] == "generation-001"
    assert data["predicted_count"] == 1
    assert data["failed_count"] == 0
    assert len(data["candidates"]) == 1


def test_generator_creation_and_rule_ordering() -> None:
    high_prior = make_rule(
        rule_id="high",
        prior_probability=0.90,
        complexity=5.0,
    )
    low_prior = make_rule(
        rule_id="low",
        prior_probability=0.20,
        complexity=0.0,
    )
    same_prior_simple = make_rule(
        rule_id="simple",
        prior_probability=0.90,
        complexity=1.0,
    )

    generator = HypothesisGenerator(
        rules=(
            low_prior,
            high_prior,
            same_prior_simple,
        ),
    )

    assert tuple(
        rule.rule_id
        for rule in generator.rules
    ) == (
        "simple",
        "high",
        "low",
    )


def test_generator_is_immutable() -> None:
    generator = HypothesisGenerator(
        rules=(make_rule(),),
    )

    with pytest.raises(FrozenInstanceError):
        generator.max_candidates = 10


def test_generator_rejects_invalid_rule_object() -> None:
    with pytest.raises(
        HypothesisGeneratorError,
        match="all rules must be CausalRule",
    ):
        HypothesisGenerator(
            rules=("rule",),
        )


def test_generator_rejects_duplicate_rule_ids() -> None:
    first = make_rule(
        rule_id="duplicate",
    )
    second = make_rule(
        rule_id="duplicate",
        hypothesis_name="Other",
    )

    with pytest.raises(
        HypothesisGeneratorError,
        match="rule_id values must be unique",
    ):
        HypothesisGenerator(
            rules=(first, second),
        )


@pytest.mark.parametrize(
    "bad_value",
    (
        0,
        -1,
    ),
)
def test_generator_rejects_nonpositive_max_candidates(
    bad_value: int,
) -> None:
    with pytest.raises(
        HypothesisGeneratorError,
        match="max_candidates must be positive",
    ):
        HypothesisGenerator(
            rules=(),
            max_candidates=bad_value,
        )


def test_generator_rejects_noninteger_max_candidates() -> None:
    with pytest.raises(
        HypothesisGeneratorError,
        match="max_candidates must be an integer",
    ):
        HypothesisGenerator(
            rules=(),
            max_candidates=1.5,
        )


def test_generator_rejects_nonboolean_fail_fast() -> None:
    with pytest.raises(
        HypothesisGeneratorError,
        match="fail_fast must be a boolean",
    ):
        HypothesisGenerator(
            rules=(),
            fail_fast=1,
        )


def test_matching_rules() -> None:
    matching = make_rule(
        rule_id="matching",
    )
    nonmatching = make_rule(
        rule_id="nonmatching",
        plane_thresholds={
            "biological": 0.50,
        },
        agent_thresholds={},
        kind_thresholds={},
        time_scale_thresholds={},
        target_thresholds={},
    )

    generator = HypothesisGenerator(
        rules=(nonmatching, matching),
    )

    result = generator.matching_rules(
        make_signature()
    )

    assert result == (matching,)


def test_matching_rules_respects_max_candidates() -> None:
    rules = tuple(
        make_rule(
            rule_id=f"rule-{index}",
            prior_probability=1.0 - index * 0.1,
        )
        for index in range(5)
    )

    generator = HypothesisGenerator(
        rules=rules,
        max_candidates=2,
    )

    assert len(
        generator.matching_rules(
            make_signature()
        )
    ) == 2


def test_matching_rules_applies_context_filters() -> None:
    mechanical = make_rule(
        rule_id="mechanical",
        initiating_plane_ids=("mechanical",),
    )
    biological = make_rule(
        rule_id="biological",
        initiating_plane_ids=("biological",),
    )

    generator = HypothesisGenerator(
        rules=(mechanical, biological),
    )

    matched = generator.matching_rules(
        make_signature(),
        active_plane_ids=("mechanical",),
    )

    assert matched == (mechanical,)


def test_seed_from_rule() -> None:
    rule = make_rule()
    observed = make_signature()

    seed = HypothesisGenerator.seed_from_rule(
        rule,
        observed,
    )

    assert seed.rule_id == rule.rule_id
    assert seed.name == rule.hypothesis_name
    assert seed.initiating_plane_ids == (
        rule.initiating_plane_ids
    )
    assert seed.prior_probability == pytest.approx(
        rule.prior_probability
    )

    assert (
        seed.metadata["source_rule_name"]
        == rule.name
    )
    assert (
        seed.metadata["observed_signature_id"]
        == observed.signature_id
    )


def test_seed_from_rule_rejects_invalid_rule() -> None:
    with pytest.raises(
        HypothesisGeneratorError,
        match="rule must be a CausalRule",
    ):
        HypothesisGenerator.seed_from_rule(
            "rule",
            make_signature(),
        )


def test_generate_seeds() -> None:
    generator = HypothesisGenerator(
        rules=(
            make_rule(
                rule_id="rule-a",
                prior_probability=0.8,
            ),
            make_rule(
                rule_id="rule-b",
                prior_probability=0.6,
            ),
        )
    )

    seeds = generator.generate_seeds(
        make_signature()
    )

    assert len(seeds) == 2
    assert all(
        isinstance(seed, HypothesisSeed)
        for seed in seeds
    )
    assert tuple(
        seed.rule_id
        for seed in seeds
    ) == (
        "rule-a",
        "rule-b",
    )


def test_generate_builds_forward_validated_hypothesis() -> None:
    observed = make_signature()
    generator = HypothesisGenerator(
        rules=(make_rule(),),
    )

    def predictor(
        seed: HypothesisSeed,
        observation: StructuralSignature,
    ) -> StructuralSignature:
        assert seed.rule_id == "rule-mechanical-impact"
        assert observation is observed

        return make_signature(
            signature_id="predicted-signature"
        )

    result = generator.generate(
        observed,
        predictor,
    )

    assert result.predicted_count == 1
    assert result.failed_count == 0

    hypothesis = result.hypotheses[0]

    assert hypothesis.status is (
        HypothesisStatus.FORWARD_VALIDATED
    )
    assert hypothesis.name == (
        "Single mechanical impact"
    )
    assert (
        hypothesis.predicted_signature.signature_id
        == "predicted-signature"
    )
    assert hypothesis.metadata["generator_version"] == (
        "1.0.0"
    )


def test_generate_passes_context_to_rule_matching() -> None:
    mechanical = make_rule(
        rule_id="mechanical",
        initiating_plane_ids=("mechanical",),
    )
    biological = make_rule(
        rule_id="biological",
        initiating_plane_ids=("biological",),
    )

    generator = HypothesisGenerator(
        rules=(mechanical, biological),
    )

    result = generator.generate(
        make_signature(),
        lambda seed, observed: observed,
        active_plane_ids=("mechanical",),
    )

    assert result.predicted_count == 1
    assert result.hypotheses[0].metadata["rule_id"] == (
        "mechanical"
    )


def test_generate_records_predictor_failure() -> None:
    generator = HypothesisGenerator(
        rules=(make_rule(),),
        fail_fast=False,
    )

    def failing_predictor(
        seed: HypothesisSeed,
        observed: StructuralSignature,
    ) -> StructuralSignature:
        raise RuntimeError("forward failure")

    result = generator.generate(
        make_signature(),
        failing_predictor,
    )

    assert result.predicted_count == 0
    assert result.failed_count == 1
    assert result.failed[0].status is (
        CandidateStatus.FAILED
    )
    assert "RuntimeError: forward failure" in (
        result.failed[0].error_message
    )


def test_generate_fail_fast_propagates_exception() -> None:
    generator = HypothesisGenerator(
        rules=(make_rule(),),
        fail_fast=True,
    )

    def failing_predictor(
        seed: HypothesisSeed,
        observed: StructuralSignature,
    ) -> StructuralSignature:
        raise RuntimeError("forward failure")

    with pytest.raises(
        RuntimeError,
        match="forward failure",
    ):
        generator.generate(
            make_signature(),
            failing_predictor,
        )


def test_generate_records_invalid_predictor_return() -> None:
    generator = HypothesisGenerator(
        rules=(make_rule(),),
        fail_fast=False,
    )

    result = generator.generate(
        make_signature(),
        lambda seed, observed: "not-a-signature",
    )

    assert result.failed_count == 1
    assert "predictor must return" in (
        result.failed[0].error_message
    )


def test_generate_rejects_noncallable_predictor() -> None:
    generator = HypothesisGenerator(
        rules=(make_rule(),),
    )

    with pytest.raises(
        HypothesisGeneratorError,
        match="predictor must be callable",
    ):
        generator.generate(
            make_signature(),
            predictor="predictor",
        )


def test_generate_with_no_matching_rules() -> None:
    rule = make_rule(
        plane_thresholds={
            "biological": 0.90,
        },
        agent_thresholds={},
        kind_thresholds={},
        time_scale_thresholds={},
        target_thresholds={},
    )

    generator = HypothesisGenerator(
        rules=(rule,),
    )

    result = generator.generate(
        make_signature(),
        lambda seed, observed: observed,
    )

    assert result.candidates == ()
    assert result.predicted_count == 0
    assert result.failed_count == 0
    assert result.metadata["matched_rule_count"] == 0


def test_generate_metadata_is_merged() -> None:
    generator = HypothesisGenerator(
        rules=(make_rule(),),
    )

    result = generator.generate(
        make_signature(),
        lambda seed, observed: observed,
        metadata={
            "experiment_id": "exp-001",
        },
    )

    assert result.metadata["generator_version"] == (
        "1.0.0"
    )
    assert result.metadata["rule_count"] == 1
    assert result.metadata["matched_rule_count"] == 1
    assert result.metadata["experiment_id"] == "exp-001"


def test_generator_round_trip() -> None:
    generator = HypothesisGenerator(
        rules=(
            make_rule(
                rule_id="rule-a",
            ),
            make_rule(
                rule_id="rule-b",
                prior_probability=0.40,
            ),
        ),
        max_candidates=10,
        fail_fast=True,
    )

    restored = HypothesisGenerator.from_dict(
        generator.to_dict()
    )

    assert restored == generator


def test_generator_serialized_version() -> None:
    data = HypothesisGenerator(
        rules=(make_rule(),),
    ).to_dict()

    assert data["version"] == "1.0.0"
    assert data["max_candidates"] == 100
    assert data["fail_fast"] is False
    assert len(data["rules"]) == 1


def test_generator_from_dict_rejects_version() -> None:
    data = HypothesisGenerator(
        rules=(),
    ).to_dict()
    data["version"] = "99.0"

    with pytest.raises(
        HypothesisGeneratorError,
        match="unsupported hypothesis generator version",
    ):
        HypothesisGenerator.from_dict(data)


def test_generator_from_dict_rejects_invalid_rules() -> None:
    data = HypothesisGenerator(
        rules=(),
    ).to_dict()
    data["rules"] = "invalid"

    with pytest.raises(
        HypothesisGeneratorError,
        match="rules must be a sequence",
    ):
        HypothesisGenerator.from_dict(data)