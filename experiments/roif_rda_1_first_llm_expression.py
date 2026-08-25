from __future__ import annotations

import json

from experiments.roif_rda_1_first_detected_difference import (
    run_first_detected_difference,
)
from roif.development.expression_gate import ExpressionGate
from roif.development.language_adapter import LanguageAdapter
from roif.development.llm_verbalizer import LLMVerbalizer
from roif.development.ollama_backend import OllamaBackend


def main() -> None:
    result, detection, profile = (
        run_first_detected_difference(
            return_internal=True,
        )
    )

    gate = ExpressionGate()

    decision = gate.decide(
        change=detection,
        difference=profile,
    )

    adapter = LanguageAdapter()

    canonical = adapter.render(
        decision
    )

    verbalizer = LLMVerbalizer(
        backend=OllamaBackend(
            model="gemma3:4b",
        )
    )

    verbalization = verbalizer.speak(
        decision,
        language="ru",
    )

    print()
    print("=== RDA-1 FIRST DETECTED DIFFERENCE ===")

    print()
    print("=== CANONICAL EXPRESSION ===")
    print()
    print(
        canonical.text
        or "[no expression]"
    )

    print()
    print("=== VERBALIZATION TRACE ===")

    print()
    print("ATTEMPT COUNT:")
    print(verbalization.attempt_count)

    print()
    print("=== FIRST CANDIDATE ===")
    print()
    print(
        verbalization.first_candidate_text
        or "[empty candidate]"
    )

    print()
    print("=== RETRY OCCURRED ===")
    print(
        "yes"
        if verbalization.retry_candidate_text is not None
        else "no"
    )

    if verbalization.retry_candidate_text is not None:
        print()
        print("=== RETRY CANDIDATE ===")
        print()
        print(
            verbalization.retry_candidate_text
            or "[empty retry candidate]"
        )

    print()
    print("=== FINAL VALIDATION STATUS ===")
    print(verbalization.validation_status)

    print()
    print("=== FINAL VALIDATION REASONS ===")

    if verbalization.validation_reasons:
        for reason in verbalization.validation_reasons:
            print(reason)
    else:
        print("[none]")

    print()
    print("=== FINAL RDA EXPRESSION ===")
    print()
    print(
        verbalization.text
        or "[no expression]"
    )

    print()
    print("=== EXPRESSION KIND ===")
    print(decision.kind.value)

    print()
    print("=== INTERNAL DETECTION ===")
    print(
        json.dumps(
            {
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
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )

    print()
    print("=== INTERNAL DIFFERENCE PROFILE ===")
    print(
        json.dumps(
            {
                "sequence_index": (
                    profile.sequence_index
                ),
                "timestamp": (
                    profile.timestamp
                ),
                "maximum_absolute_deviation": (
                    profile.maximum_absolute_deviation
                ),
                "l2_deviation_norm": (
                    profile.l2_deviation_norm
                ),
                "channel_count": len(
                    profile.channel_deviations
                ),
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )

    print()
    print("=== IDENTITY CHECK ===")
    print(
        json.dumps(
            result["identity_check"],
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )

    print()
    print("=== SOURCE ===")
    print(
        json.dumps(
            {
                "benchmark": (
                    result["benchmark"]
                ),
                "developmental_stage": (
                    result["developmental_stage"]
                ),
                "experience": (
                    result["experience"]
                ),
                "run_sha256": (
                    result["run_sha256"]
                ),
                "verbalizer": (
                    verbalization.model_name
                ),
                "used_llm": (
                    verbalization.used_llm
                ),
                "validation_status": (
                    verbalization.validation_status
                ),
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
