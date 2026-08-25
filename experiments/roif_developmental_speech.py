from __future__ import annotations

import json
from typing import Any

from experiments.roif_rda_0_first_self_detected_change import (
    run_first_self_detected_change,
)
from experiments.roif_rda_2_first_endogenous_grouping import (
    run_first_endogenous_grouping,
)
from experiments.roif_rda_3_first_dimensional_growth import (
    run_first_dimensional_growth,
)
from roif.development.expression_gate import ExpressionGate
from roif.development.language_adapter import LanguageAdapter
from roif.development.llm_verbalizer import LLMVerbalizer
from roif.development.ollama_backend import OllamaBackend


MODEL_NAME = "gemma3:4b"
LANGUAGE = "ru"


def _verbalize(
    *,
    gate: ExpressionGate,
    adapter: LanguageAdapter,
    verbalizer: LLMVerbalizer,
    gate_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """
    Convert one already-existing internal developmental state into
    validated language.

    This function does not infer developmental facts from benchmark JSON.
    The ExpressionGate receives the real internal object produced by the
    corresponding developmental mechanism.
    """

    decision = gate.decide(
        **gate_kwargs
    )

    canonical = adapter.render(
        decision
    )

    verbalization = verbalizer.speak(
        decision,
        language=LANGUAGE,
    )

    return {
        "expression_kind": decision.kind.value,
        "canonical_expression": (
            canonical.text
            or ""
        ),
        "final_expression": (
            verbalization.text
            or ""
        ),
        "validation_status": (
            verbalization.validation_status
        ),
        "validation_reasons": list(
            verbalization.validation_reasons
        ),
        "attempt_count": (
            verbalization.attempt_count
        ),
        "first_candidate": (
            verbalization.first_candidate_text
        ),
        "retry_candidate": (
            verbalization.retry_candidate_text
        ),
        "retry_occurred": (
            verbalization.retry_candidate_text
            is not None
        ),
        "used_llm": (
            verbalization.used_llm
        ),
        "verbalizer": (
            verbalization.model_name
        ),
    }


def main() -> None:
    # ---------------------------------------------------------
    # Shared expression layer.
    #
    # The LLM is downstream of developmental computation.
    # It receives ExpressionDecision objects, not simulator
    # ground truth and not benchmark summaries.
    # ---------------------------------------------------------

    gate = ExpressionGate()
    adapter = LanguageAdapter()

    verbalizer = LLMVerbalizer(
        backend=OllamaBackend(
            model=MODEL_NAME,
        )
    )

    # ---------------------------------------------------------
    # RDA-0
    #
    # Real internal source:
    #     ChangeDetection
    # ---------------------------------------------------------

    rda0_result, detection = (
        run_first_self_detected_change(
            return_internal=True,
        )
    )

    rda0_speech = _verbalize(
        gate=gate,
        adapter=adapter,
        verbalizer=verbalizer,
        gate_kwargs={
            "change": detection,
        },
    )

    rda0 = {
        "stage": "RDA-0",
        "experience": (
            rda0_result["experience"]
        ),
        "benchmark": (
            rda0_result["benchmark"]
        ),
        "run_sha256": (
            rda0_result["run_sha256"]
        ),
        "internal_source_type": (
            type(detection).__name__
        ),
        "internal_state": {
            "sequence_index": (
                detection.sequence_index
            ),
            "timestamp": (
                detection.timestamp
            ),
            "deviation_score": (
                detection.deviation_score
            ),
            "changed": (
                detection.changed
            ),
        },
        "speech": rda0_speech,
    }

    # ---------------------------------------------------------
    # RDA-2
    #
    # Real internal source:
    #     GroupingResult
    # ---------------------------------------------------------

    rda2_result, grouping = (
        run_first_endogenous_grouping(
            return_internal=True,
        )
    )

    rda2_speech = _verbalize(
        gate=gate,
        adapter=adapter,
        verbalizer=verbalizer,
        gate_kwargs={
            "grouping": grouping,
        },
    )

    strongest_group_strength = (
        None
        if not grouping.groups
        else max(
            group.internal_relation_strength
            for group in grouping.groups
        )
    )

    rda2 = {
        "stage": "RDA-2",
        "experience": (
            rda2_result["experience"]
        ),
        "benchmark": (
            rda2_result["benchmark"]
        ),
        "run_sha256": (
            rda2_result["run_sha256"]
        ),
        "internal_source_type": (
            type(grouping).__name__
        ),
        "internal_state": {
            "channel_count": len(
                grouping.channel_names
            ),
            "group_count": len(
                grouping.groups
            ),
            "relation_count": len(
                grouping.relations
            ),
            "strongest_group_strength": (
                strongest_group_strength
            ),
        },
        "speech": rda2_speech,
    }

    # ---------------------------------------------------------
    # RDA-3
    #
    # Real internal source:
    #     DimensionalGrowthAssessment
    # ---------------------------------------------------------

    rda3_result, assessment = (
        run_first_dimensional_growth(
            return_internal=True,
        )
    )

    rda3_speech = _verbalize(
        gate=gate,
        adapter=adapter,
        verbalizer=verbalizer,
        gate_kwargs={
            "dimensional_growth": assessment,
        },
    )

    rda3 = {
        "stage": "RDA-3",
        "experience": (
            rda3_result["experience"]
        ),
        "benchmark": (
            rda3_result["benchmark"]
        ),
        "run_sha256": (
            rda3_result["run_sha256"]
        ),
        "internal_source_type": (
            type(assessment).__name__
        ),
        "internal_state": {
            "growth_supported": (
                assessment.growth_supported
            ),
            "represented_dimension": (
                assessment.represented_dimension
            ),
            "persistent_residual_dimension_count": (
                assessment
                .persistent_residual_dimension_count
            ),
            "residual_variance_fraction": (
                assessment.residual_variance_fraction
            ),
            "residual_effective_rank": (
                assessment.residual_effective_rank
            ),
            "residual_numerical_rank": (
                assessment.residual_numerical_rank
            ),
            "residual_participation_ratio": (
                assessment.residual_participation_ratio
            ),
        },
        "speech": rda3_speech,
    }

    stages = [
        rda0,
        rda2,
        rda3,
    ]

    # ---------------------------------------------------------
    # Human-readable developmental speech sequence.
    # ---------------------------------------------------------

    print()
    print("==============================================")
    print("ROIF DEVELOPMENTAL SPEECH")
    print("==============================================")

    for stage in stages:
        print()
        print(
            "----------------------------------------------"
        )
        print(stage["stage"])
        print(
            "----------------------------------------------"
        )

        print()
        print("INTERNAL SOURCE:")
        print(stage["internal_source_type"])

        print()
        print("EXPRESSION KIND:")
        print(
            stage["speech"]["expression_kind"]
        )

        print()
        print("CANONICAL:")
        print(
            stage["speech"]["canonical_expression"]
            or "[no expression]"
        )

        print()
        print("ROIF:")
        print(
            stage["speech"]["final_expression"]
            or "[no expression]"
        )

        print()
        print("VALIDATION:")
        print(
            stage["speech"]["validation_status"]
        )

        if stage["speech"]["validation_reasons"]:
            print("REASONS:")
            for reason in (
                stage["speech"][
                    "validation_reasons"
                ]
            ):
                print(reason)

        print()
        print("RUN SHA256:")
        print(stage["run_sha256"])

    # ---------------------------------------------------------
    # Machine-readable audit.
    # ---------------------------------------------------------

    audit = {
        "sequence": (
            "roif_developmental_speech_v1"
        ),
        "language": LANGUAGE,
        "model": MODEL_NAME,
        "stage_count": len(stages),
        "stages": stages,
        "architectural_boundary": {
            "speech_reads_real_internal_objects": True,
            "speech_reads_benchmark_ground_truth": False,
            "llm_creates_developmental_state": False,
            "llm_role": (
                "validated_verbalization_only"
            ),
        },
        "continuity_claim": (
            "independent_frozen_developmental_benchmarks"
        ),
    }

    print()
    print("==============================================")
    print("AUDIT")
    print("==============================================")
    print()

    print(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
