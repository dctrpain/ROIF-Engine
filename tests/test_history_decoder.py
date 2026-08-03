from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from roif.history.history_decoder import (
    DecodeStatus,
    DecoderWeights,
    HistoryDecodeResult,
    HistoryDecoder,
    HistoryDecoderError,
    HistoryHypothesis,
    HypothesisScore,
    HypothesisStatus,
)
from roif.history.structural_signature import (
    SignatureComparison,
    SignatureDistanceWeights,
    StructuralSignature,
)


def make_signature(
    signature_id: str = "signature-observed",
    **overrides,
) -> StructuralSignature:
    data = {
        "signature_id": signature_id,
        "source_pattern_id": f"pattern-{signature_id}",
        "label": signature_id,
        "plane_profile": {
            "mechanical": 0.80,
            "thermal": 0.20,
        },
        "agent_profile": {
            "steel_object": 0.75,
            "environment": 0.25,
        },
        "kind_profile": {
            "damage": 0.80,
            "remodeling": 0.20,
        },
        "time_scale_profile": {
            "medium": 0.80,
            "slow": 0.20,
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
        "harmful_fraction": 0.80,
        "beneficial_fraction": 0.20,
        "mixed_fraction": 0.0,
        "metadata": {
            "source": "test",
        },
    }

    data.update(overrides)
    return StructuralSignature(**data)


def make_hypothesis(
    hypothesis_id: str = "hypothesis-a",
    *,
    predicted_signature: StructuralSignature | None = None,
    **overrides,
) -> HistoryHypothesis:
    if predicted_signature is None:
        predicted_signature = make_signature(
            signature_id=f"predicted-{hypothesis_id}"
        )

    data = {
        "hypothesis_id": hypothesis_id,
        "name": f"Hypothesis {hypothesis_id}",
        "description": "Candidate structural history.",
        "predicted_signature": predicted_signature,
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
        "custom_penalty": 0.0,
        "status": HypothesisStatus.FORWARD_VALIDATED,
        "metadata": {
            "model": "forward-test",
        },
    }

    data.update(overrides)
    return HistoryHypothesis(**data)


def make_comparison(
    *,
    left_signature_id: str = "observed",
    right_signature_id: str = "predicted",
    distance: float = 0.20,
) -> SignatureComparison:
    return SignatureComparison(
        left_signature_id=left_signature_id,
        right_signature_id=right_signature_id,
        distance=distance,
        similarity=1.0 - distance,
        components={
            "plane_profile": distance,
            "capacity": distance,
        },
        weights=SignatureDistanceWeights(),
    )


def make_score(
    hypothesis_id: str = "hypothesis-a",
    *,
    rank: int = 1,
    normalized_score: float = 0.80,
    raw_score: float | None = None,
) -> HypothesisScore:
    if raw_score is None:
        raw_score = normalized_score

    hypothesis = make_hypothesis(
        hypothesis_id=hypothesis_id
    )

    return HypothesisScore(
        hypothesis=hypothesis,
        comparison=make_comparison(
            right_signature_id=(
                hypothesis.predicted_signature.signature_id
            )
        ),
        fit_score=0.85,
        prior_score=0.60,
        validation_score=0.90,
        complexity_penalty=1.0 / 6.0,
        custom_penalty=0.0,
        raw_score=raw_score,
        normalized_score=normalized_score,
        rank=rank,
    )


def test_decoder_weights_creation() -> None:
    weights = DecoderWeights()

    assert weights.signature_fit == pytest.approx(1.0)
    assert weights.prior == pytest.approx(0.25)
    assert weights.validation == pytest.approx(0.50)
    assert weights.complexity == pytest.approx(0.20)
    assert weights.custom_penalty == pytest.approx(0.20)

    assert weights.total_positive_weight == pytest.approx(
        1.75
    )


def test_decoder_weights_are_immutable() -> None:
    weights = DecoderWeights()

    with pytest.raises(FrozenInstanceError):
        weights.prior = 1.0


def test_decoder_weights_round_trip() -> None:
    weights = DecoderWeights(
        signature_fit=2.0,
        prior=0.5,
        validation=1.0,
        complexity=0.3,
        custom_penalty=0.4,
    )

    restored = DecoderWeights.from_dict(
        weights.to_dict()
    )

    assert restored == weights


@pytest.mark.parametrize(
    "field_name",
    (
        "signature_fit",
        "prior",
        "validation",
        "complexity",
        "custom_penalty",
    ),
)
def test_decoder_weights_reject_negative_values(
    field_name: str,
) -> None:
    with pytest.raises(
        HistoryDecoderError,
        match=field_name,
    ):
        DecoderWeights(
            **{
                field_name: -0.01,
            }
        )


def test_decoder_weights_require_positive_scoring_weight() -> None:
    with pytest.raises(
        HistoryDecoderError,
        match="at least one positive decoder weight",
    ):
        DecoderWeights(
            signature_fit=0.0,
            prior=0.0,
            validation=0.0,
        )


def test_decoder_weights_from_dict_rejects_invalid_data() -> None:
    with pytest.raises(
        HistoryDecoderError,
        match="must be a mapping",
    ):
        DecoderWeights.from_dict("weights")


def test_history_hypothesis_creation() -> None:
    hypothesis = make_hypothesis()

    assert hypothesis.hypothesis_id == "hypothesis-a"
    assert hypothesis.name == "Hypothesis hypothesis-a"

    assert isinstance(
        hypothesis.predicted_signature,
        StructuralSignature,
    )

    assert hypothesis.initiating_plane_ids == (
        "mechanical",
    )
    assert hypothesis.initiating_agent_ids == (
        "steel_object",
    )
    assert hypothesis.candidate_root_ids == (
        "element:A",
    )

    assert hypothesis.prior_probability == pytest.approx(
        0.60
    )
    assert hypothesis.validation_score == pytest.approx(
        0.90
    )
    assert hypothesis.complexity == pytest.approx(1.0)

    assert (
        hypothesis.status
        is HypothesisStatus.FORWARD_VALIDATED
    )


def test_hypothesis_accepts_string_status() -> None:
    hypothesis = make_hypothesis(
        status="ready",
    )

    assert hypothesis.status is HypothesisStatus.READY


def test_hypothesis_is_immutable() -> None:
    hypothesis = make_hypothesis()

    with pytest.raises(FrozenInstanceError):
        hypothesis.name = "Changed"


def test_hypothesis_metadata_is_read_only() -> None:
    hypothesis = make_hypothesis()

    with pytest.raises(TypeError):
        hypothesis.metadata["new"] = "value"


def test_hypothesis_ids_are_deduplicated() -> None:
    hypothesis = make_hypothesis(
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

    assert hypothesis.initiating_plane_ids == (
        "mechanical",
        "thermal",
    )
    assert hypothesis.initiating_agent_ids == (
        "steel",
        "wood",
    )
    assert hypothesis.candidate_root_ids == (
        "A",
        "B",
    )


@pytest.mark.parametrize(
    ("status", "expected"),
    (
        (HypothesisStatus.READY, True),
        (HypothesisStatus.FORWARD_VALIDATED, True),
        (HypothesisStatus.INCOMPLETE, True),
        (HypothesisStatus.REJECTED, False),
        (HypothesisStatus.FAILED, False),
    ),
)
def test_hypothesis_eligibility(
    status: HypothesisStatus,
    expected: bool,
) -> None:
    hypothesis = make_hypothesis(
        status=status,
    )

    assert hypothesis.is_eligible is expected


@pytest.mark.parametrize(
    "bad_value",
    (
        -0.01,
        1.01,
        float("inf"),
        float("nan"),
    ),
)
def test_hypothesis_rejects_invalid_prior(
    bad_value: float,
) -> None:
    with pytest.raises(
        HistoryDecoderError,
        match="prior_probability",
    ):
        make_hypothesis(
            prior_probability=bad_value,
        )


@pytest.mark.parametrize(
    "bad_value",
    (
        -0.01,
        1.01,
        float("inf"),
        float("nan"),
    ),
)
def test_hypothesis_rejects_invalid_validation_score(
    bad_value: float,
) -> None:
    with pytest.raises(
        HistoryDecoderError,
        match="validation_score",
    ):
        make_hypothesis(
            validation_score=bad_value,
        )


@pytest.mark.parametrize(
    "bad_value",
    (
        -0.01,
        1.01,
        float("inf"),
        float("nan"),
    ),
)
def test_hypothesis_rejects_invalid_custom_penalty(
    bad_value: float,
) -> None:
    with pytest.raises(
        HistoryDecoderError,
        match="custom_penalty",
    ):
        make_hypothesis(
            custom_penalty=bad_value,
        )


def test_hypothesis_rejects_negative_complexity() -> None:
    with pytest.raises(
        HistoryDecoderError,
        match="complexity",
    ):
        make_hypothesis(
            complexity=-1.0,
        )


def test_hypothesis_rejects_invalid_signature() -> None:
    with pytest.raises(
        HistoryDecoderError,
        match="predicted_signature",
    ):
        make_hypothesis(
            predicted_signature="signature",
        )


def test_hypothesis_rejects_invalid_status() -> None:
    with pytest.raises(
        HistoryDecoderError,
        match="unsupported hypothesis status",
    ):
        make_hypothesis(
            status="unsupported",
        )


def test_hypothesis_round_trip() -> None:
    hypothesis = make_hypothesis()

    restored = HistoryHypothesis.from_dict(
        hypothesis.to_dict()
    )

    assert restored == hypothesis
    assert restored is not hypothesis
    assert (
        restored.predicted_signature
        is not hypothesis.predicted_signature
    )


def test_hypothesis_serialized_version() -> None:
    data = make_hypothesis().to_dict()

    assert data["version"] == "1.0.0"
    assert data["hypothesis_id"] == "hypothesis-a"
    assert data["status"] == "forward_validated"
    assert "predicted_signature" in data


def test_hypothesis_from_dict_rejects_version() -> None:
    data = make_hypothesis().to_dict()
    data["version"] = "99.0"

    with pytest.raises(
        HistoryDecoderError,
        match="unsupported history hypothesis version",
    ):
        HistoryHypothesis.from_dict(data)


def test_hypothesis_from_dict_rejects_missing_signature() -> None:
    data = make_hypothesis().to_dict()
    data.pop("predicted_signature")

    with pytest.raises(
        HistoryDecoderError,
        match="predicted_signature is missing",
    ):
        HistoryHypothesis.from_dict(data)


def test_hypothesis_score_creation() -> None:
    score = make_score()

    assert score.hypothesis_id == "hypothesis-a"
    assert score.name == "Hypothesis hypothesis-a"
    assert score.rank == 1
    assert score.normalized_score == pytest.approx(0.80)


def test_hypothesis_score_is_immutable() -> None:
    score = make_score()

    with pytest.raises(FrozenInstanceError):
        score.rank = 2


def test_hypothesis_score_with_rank() -> None:
    score = make_score(
        rank=0,
    )

    ranked = score.with_rank(3)

    assert score.rank == 0
    assert ranked.rank == 3
    assert ranked.hypothesis == score.hypothesis


@pytest.mark.parametrize(
    "bad_rank",
    (
        0,
        -1,
        1.5,
    ),
)
def test_hypothesis_score_with_rank_rejects_invalid(
    bad_rank,
) -> None:
    score = make_score(
        rank=0,
    )

    with pytest.raises(
        HistoryDecoderError,
        match="rank must be a positive integer",
    ):
        score.with_rank(bad_rank)


def test_hypothesis_score_rejects_invalid_hypothesis() -> None:
    with pytest.raises(
        HistoryDecoderError,
        match="hypothesis must be",
    ):
        HypothesisScore(
            hypothesis="hypothesis",
            comparison=make_comparison(),
            fit_score=0.8,
            prior_score=0.5,
            validation_score=0.8,
            complexity_penalty=0.1,
            custom_penalty=0.0,
            raw_score=0.7,
            normalized_score=0.7,
        )


def test_hypothesis_score_rejects_invalid_comparison() -> None:
    hypothesis = make_hypothesis()

    with pytest.raises(
        HistoryDecoderError,
        match="comparison must be",
    ):
        HypothesisScore(
            hypothesis=hypothesis,
            comparison="comparison",
            fit_score=0.8,
            prior_score=0.5,
            validation_score=0.8,
            complexity_penalty=0.1,
            custom_penalty=0.0,
            raw_score=0.7,
            normalized_score=0.7,
        )


def test_hypothesis_score_to_dict() -> None:
    score = make_score()
    data = score.to_dict()

    assert data["hypothesis_id"] == "hypothesis-a"
    assert data["rank"] == 1
    assert data["normalized_score"] == pytest.approx(
        0.80
    )
    assert "comparison" in data


def test_decode_result_creation() -> None:
    first = make_score(
        "hypothesis-a",
        rank=1,
        normalized_score=0.90,
    )
    second = make_score(
        "hypothesis-b",
        rank=2,
        normalized_score=0.70,
    )

    result = HistoryDecodeResult(
        result_id="result-001",
        observed_signature_id="observed",
        status=DecodeStatus.COMPLETED,
        rankings=(first, second),
        confidence=0.85,
        ambiguity=0.80,
        selection_margin=0.20,
        metadata={
            "test": True,
        },
    )

    assert result.result_id == "result-001"
    assert result.best == first
    assert result.alternatives == (second,)
    assert result.is_ambiguous is False


def test_decode_result_is_immutable() -> None:
    result = HistoryDecodeResult(
        observed_signature_id="observed",
        status=DecodeStatus.NO_CANDIDATES,
    )

    with pytest.raises(FrozenInstanceError):
        result.status = DecodeStatus.COMPLETED


def test_decode_result_metadata_is_read_only() -> None:
    result = HistoryDecodeResult(
        observed_signature_id="observed",
        status=DecodeStatus.NO_CANDIDATES,
    )

    with pytest.raises(TypeError):
        result.metadata["new"] = "value"


def test_decode_result_top() -> None:
    rankings = (
        make_score("a", rank=1),
        make_score("b", rank=2),
        make_score("c", rank=3),
    )

    result = HistoryDecodeResult(
        observed_signature_id="observed",
        status=DecodeStatus.COMPLETED,
        rankings=rankings,
    )

    assert result.top(2) == rankings[:2]
    assert result.top(0) == ()


def test_decode_result_top_rejects_invalid_count() -> None:
    result = HistoryDecodeResult(
        observed_signature_id="observed",
        status=DecodeStatus.NO_CANDIDATES,
    )

    with pytest.raises(
        HistoryDecoderError,
        match="count must be an integer",
    ):
        result.top(1.5)

    with pytest.raises(
        HistoryDecoderError,
        match="count must be non-negative",
    ):
        result.top(-1)


def test_decode_result_rejects_nonconsecutive_ranks() -> None:
    rankings = (
        make_score("a", rank=1),
        make_score("b", rank=3),
    )

    with pytest.raises(
        HistoryDecoderError,
        match="ranking positions must be consecutive",
    ):
        HistoryDecodeResult(
            observed_signature_id="observed",
            status=DecodeStatus.COMPLETED,
            rankings=rankings,
        )


def test_decode_result_to_dict() -> None:
    result = HistoryDecodeResult(
        result_id="result-001",
        observed_signature_id="observed",
        status=DecodeStatus.NO_CANDIDATES,
    )

    data = result.to_dict()

    assert data["version"] == "1.0.0"
    assert data["result_id"] == "result-001"
    assert data["status"] == "no_candidates"
    assert data["rankings"] == []


def test_history_decoder_creation() -> None:
    decoder = HistoryDecoder()

    assert decoder.ambiguity_threshold == pytest.approx(
        0.05
    )
    assert decoder.complexity_scale == pytest.approx(
        5.0
    )
    assert decoder.minimum_score == pytest.approx(0.0)


def test_history_decoder_is_immutable() -> None:
    decoder = HistoryDecoder()

    with pytest.raises(FrozenInstanceError):
        decoder.minimum_score = 0.5


def test_decoder_rejects_invalid_weights() -> None:
    with pytest.raises(
        HistoryDecoderError,
        match="decoder_weights must be",
    ):
        HistoryDecoder(
            decoder_weights="weights",
        )

    with pytest.raises(
        HistoryDecoderError,
        match="signature_weights must be",
    ):
        HistoryDecoder(
            signature_weights="weights",
        )


def test_decoder_rejects_zero_complexity_scale() -> None:
    with pytest.raises(
        HistoryDecoderError,
        match="complexity_scale must be greater than zero",
    ):
        HistoryDecoder(
            complexity_scale=0.0,
        )


@pytest.mark.parametrize(
    "field_name",
    (
        "ambiguity_threshold",
        "minimum_score",
    ),
)
def test_decoder_rejects_invalid_unit_interval(
    field_name: str,
) -> None:
    with pytest.raises(
        HistoryDecoderError,
        match=field_name,
    ):
        HistoryDecoder(
            **{
                field_name: 1.1,
            }
        )


def test_score_identical_hypothesis() -> None:
    observed = make_signature()
    hypothesis = make_hypothesis(
        predicted_signature=make_signature(
            signature_id="predicted"
        ),
        complexity=0.0,
        custom_penalty=0.0,
    )

    decoder = HistoryDecoder()
    score = decoder.score_hypothesis(
        observed,
        hypothesis,
    )

    assert score.fit_score == pytest.approx(1.0)
    assert score.normalized_score > 0.0
    assert score.rank == 0


def test_complexity_reduces_hypothesis_score() -> None:
    observed = make_signature()

    simple = make_hypothesis(
        hypothesis_id="simple",
        complexity=0.0,
    )
    complex_hypothesis = make_hypothesis(
        hypothesis_id="complex",
        complexity=100.0,
    )

    decoder = HistoryDecoder()

    simple_score = decoder.score_hypothesis(
        observed,
        simple,
    )
    complex_score = decoder.score_hypothesis(
        observed,
        complex_hypothesis,
    )

    assert (
        simple_score.normalized_score
        > complex_score.normalized_score
    )


def test_custom_penalty_reduces_score() -> None:
    observed = make_signature()

    unpenalized = make_hypothesis(
        hypothesis_id="unpenalized",
        custom_penalty=0.0,
    )
    penalized = make_hypothesis(
        hypothesis_id="penalized",
        custom_penalty=1.0,
    )

    decoder = HistoryDecoder()

    assert (
        decoder.score_hypothesis(
            observed,
            unpenalized,
        ).normalized_score
        >
        decoder.score_hypothesis(
            observed,
            penalized,
        ).normalized_score
    )


def test_score_rejects_failed_hypothesis() -> None:
    decoder = HistoryDecoder()

    with pytest.raises(
        HistoryDecoderError,
        match="cannot be scored",
    ):
        decoder.score_hypothesis(
            make_signature(),
            make_hypothesis(
                status=HypothesisStatus.FAILED,
            ),
        )


def test_decode_no_candidates() -> None:
    observed = make_signature()

    result = HistoryDecoder().decode(
        observed,
        (),
    )

    assert result.status is DecodeStatus.NO_CANDIDATES
    assert result.rankings == ()
    assert result.best is None
    assert result.confidence == pytest.approx(0.0)
    assert result.ambiguity == pytest.approx(1.0)


def test_decode_no_valid_candidates() -> None:
    hypotheses = (
        make_hypothesis(
            "rejected",
            status=HypothesisStatus.REJECTED,
        ),
        make_hypothesis(
            "failed",
            status=HypothesisStatus.FAILED,
        ),
    )

    result = HistoryDecoder().decode(
        make_signature(),
        hypotheses,
    )

    assert (
        result.status
        is DecodeStatus.NO_VALID_CANDIDATES
    )
    assert result.rankings == ()


def test_decode_ranks_best_matching_signature_first() -> None:
    observed = make_signature()

    matching = make_hypothesis(
        hypothesis_id="matching",
        predicted_signature=make_signature(
            signature_id="matching-signature"
        ),
        complexity=0.0,
    )

    different = make_hypothesis(
        hypothesis_id="different",
        predicted_signature=make_signature(
            signature_id="different-signature",
            plane_profile={
                "biological": 1.0,
            },
            progression_index=0.1,
            adaptation_index=0.9,
            total_capacity_loss=0.05,
            total_capacity_gain=0.30,
            net_capacity_effect=0.25,
            harmful_fraction=0.1,
            beneficial_fraction=0.9,
        ),
        complexity=0.0,
    )

    result = HistoryDecoder().decode(
        observed,
        (
            different,
            matching,
        ),
    )

    assert result.best is not None
    assert result.best.hypothesis_id == "matching"
    assert result.rankings[0].rank == 1
    assert result.rankings[1].rank == 2


def test_decode_identical_candidates_is_ambiguous() -> None:
    observed = make_signature()

    first = make_hypothesis(
        hypothesis_id="first",
        complexity=0.0,
    )
    second = make_hypothesis(
        hypothesis_id="second",
        complexity=0.0,
    )

    result = HistoryDecoder(
        ambiguity_threshold=0.05,
    ).decode(
        observed,
        (first, second),
    )

    assert result.status is DecodeStatus.AMBIGUOUS
    assert result.is_ambiguous is True
    assert result.selection_margin == pytest.approx(
        0.0
    )
    assert result.ambiguity == pytest.approx(1.0)


def test_decode_single_candidate_is_completed() -> None:
    result = HistoryDecoder().decode(
        make_signature(),
        (
            make_hypothesis(
                complexity=0.0,
            ),
        ),
    )

    assert result.status is DecodeStatus.COMPLETED
    assert result.best is not None
    assert result.selection_margin == pytest.approx(
        result.best.normalized_score
    )


def test_decode_filters_by_minimum_score() -> None:
    weak = make_hypothesis(
        hypothesis_id="weak",
        predicted_signature=make_signature(
            signature_id="weak-signature",
            plane_profile={
                "biological": 1.0,
            },
        ),
        prior_probability=0.0,
        validation_score=0.0,
        complexity=1000.0,
        custom_penalty=1.0,
    )

    decoder = HistoryDecoder(
        minimum_score=0.90,
    )

    result = decoder.decode(
        make_signature(),
        (weak,),
    )

    assert (
        result.status
        is DecodeStatus.NO_VALID_CANDIDATES
    )


def test_decode_ignores_rejected_candidates() -> None:
    valid = make_hypothesis(
        hypothesis_id="valid",
    )
    rejected = make_hypothesis(
        hypothesis_id="rejected",
        status=HypothesisStatus.REJECTED,
    )

    result = HistoryDecoder().decode(
        make_signature(),
        (rejected, valid),
    )

    assert len(result.rankings) == 1
    assert result.best is not None
    assert result.best.hypothesis_id == "valid"

    assert result.metadata["candidate_count"] == 2
    assert result.metadata["eligible_count"] == 1


def test_decode_rejects_duplicate_hypothesis_ids() -> None:
    first = make_hypothesis(
        hypothesis_id="duplicate",
    )
    second = make_hypothesis(
        hypothesis_id="duplicate",
        name="Other duplicate",
    )

    with pytest.raises(
        HistoryDecoderError,
        match="hypothesis_id values must be unique",
    ):
        HistoryDecoder().decode(
            make_signature(),
            (first, second),
        )


def test_decode_rejects_invalid_hypothesis_object() -> None:
    with pytest.raises(
        HistoryDecoderError,
        match="all hypotheses must be",
    ):
        HistoryDecoder().decode(
            make_signature(),
            ("hypothesis",),
        )


def test_decode_metadata_is_merged() -> None:
    result = HistoryDecoder().decode(
        make_signature(),
        (
            make_hypothesis(),
        ),
        metadata={
            "experiment_id": "exp-001",
        },
    )

    assert result.metadata["decoder_version"] == "1.0.0"
    assert result.metadata["candidate_count"] == 1
    assert result.metadata["experiment_id"] == "exp-001"


def test_softmax_probabilities_sum_to_one() -> None:
    result = HistoryDecoder().decode(
        make_signature(),
        (
            make_hypothesis(
                hypothesis_id="a",
            ),
            make_hypothesis(
                hypothesis_id="b",
                prior_probability=0.2,
            ),
        ),
    )

    probabilities = (
        HistoryDecoder().softmax_probabilities(
            result
        )
    )

    assert sum(probabilities.values()) == pytest.approx(
        1.0
    )
    assert set(probabilities) == {
        "a",
        "b",
    }


def test_softmax_probabilities_are_read_only() -> None:
    decoder = HistoryDecoder()
    result = decoder.decode(
        make_signature(),
        (
            make_hypothesis(),
        ),
    )

    probabilities = decoder.softmax_probabilities(
        result
    )

    with pytest.raises(TypeError):
        probabilities["new"] = 0.5


def test_softmax_empty_result() -> None:
    decoder = HistoryDecoder()

    result = decoder.decode(
        make_signature(),
        (),
    )

    assert decoder.softmax_probabilities(
        result
    ) == {}


def test_softmax_temperature_changes_distribution() -> None:
    decoder = HistoryDecoder()

    result = decoder.decode(
        make_signature(),
        (
            make_hypothesis(
                hypothesis_id="strong",
                prior_probability=1.0,
            ),
            make_hypothesis(
                hypothesis_id="weak",
                prior_probability=0.0,
                validation_score=0.2,
            ),
        ),
    )

    cold = decoder.softmax_probabilities(
        result,
        temperature=0.1,
    )
    warm = decoder.softmax_probabilities(
        result,
        temperature=10.0,
    )

    assert cold["strong"] > warm["strong"]


def test_softmax_rejects_invalid_temperature() -> None:
    decoder = HistoryDecoder()

    result = decoder.decode(
        make_signature(),
        (
            make_hypothesis(),
        ),
    )

    with pytest.raises(
        HistoryDecoderError,
        match="temperature must be greater than zero",
    ):
        decoder.softmax_probabilities(
            result,
            temperature=0.0,
        )


def test_softmax_rejects_invalid_result() -> None:
    with pytest.raises(
        HistoryDecoderError,
        match="result must be a HistoryDecodeResult",
    ):
        HistoryDecoder().softmax_probabilities(
            "result"
        )


def test_decoder_round_trip() -> None:
    decoder = HistoryDecoder(
        decoder_weights=DecoderWeights(
            signature_fit=2.0,
            prior=0.5,
            validation=1.0,
            complexity=0.4,
            custom_penalty=0.3,
        ),
        signature_weights=SignatureDistanceWeights(
            plane_profile=2.0,
            capacity=3.0,
        ),
        ambiguity_threshold=0.10,
        complexity_scale=8.0,
        minimum_score=0.20,
    )

    restored = HistoryDecoder.from_dict(
        decoder.to_dict()
    )

    assert restored == decoder


def test_decoder_serialized_version() -> None:
    data = HistoryDecoder().to_dict()

    assert data["version"] == "1.0.0"
    assert "decoder_weights" in data
    assert "signature_weights" in data


def test_decoder_from_dict_rejects_version() -> None:
    data = HistoryDecoder().to_dict()
    data["version"] = "99.0"

    with pytest.raises(
        HistoryDecoderError,
        match="unsupported history decoder version",
    ):
        HistoryDecoder.from_dict(data)


def test_decoder_from_dict_rejects_invalid_weights() -> None:
    data = HistoryDecoder().to_dict()
    data["decoder_weights"] = "invalid"

    with pytest.raises(
        HistoryDecoderError,
        match="decoder_weights must be a mapping",
    ):
        HistoryDecoder.from_dict(data)


def test_decoder_from_dict_rejects_invalid_signature_weights() -> None:
    data = HistoryDecoder().to_dict()
    data["signature_weights"] = "invalid"

    with pytest.raises(
        HistoryDecoderError,
        match="signature_weights must be a mapping",
    ):
        HistoryDecoder.from_dict(data)