from __future__ import annotations

import json

from experiments.roif_rda_0_first_self_detected_change import (
    run_first_self_detected_change,
)
from roif.development.expression_gate import ExpressionGate
from roif.development.language_adapter import LanguageAdapter
from roif.development.llm_verbalizer import LLMVerbalizer
from roif.development.ollama_backend import OllamaBackend


def main() -> None:
    result, detection = run_first_self_detected_change(
        return_internal=True,
    )

    gate = ExpressionGate()

    decision = gate.decide(
        change=detection,
    )

    canonical_adapter = LanguageAdapter()
    canonical = canonical_adapter.render(decision)

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
    print("=== RDA-0 FIRST SELF-DETECTED CHANGE ===")

    print()
    print("=== CANONICAL EXPRESSION ===")
    print()
    print(canonical.text or "[no expression]")

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

    if verbalization.retry_candidate_text is not None:
        print()
        print("=== RETRY OCCURRED ===")
        print("yes")

        print()
        print("=== RETRY CANDIDATE ===")
        print()
        print(
            verbalization.retry_candidate_text
            or "[empty retry candidate]"
        )
    else:
        print()
        print("=== RETRY OCCURRED ===")
        print("no")

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
    print(verbalization.text or "[no expression]")

    print()
    print("=== EXPRESSION KIND ===")
    print(decision.kind.value)

    print()
    print("=== INTERNAL CHANGE DETECTION ===")
    print(
        json.dumps(
            {
                "sequence_index": detection.sequence_index,
                "timestamp": detection.timestamp,
                "deviation_score": detection.deviation_score,
                "changed": detection.changed,
            },
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
                "benchmark": result["benchmark"],
                "developmental_stage": result[
                    "developmental_stage"
                ],
                "experience": result["experience"],
                "run_sha256": result["run_sha256"],
                "used_llm": verbalization.used_llm,
                "validation_status": (
                    verbalization.validation_status
                ),
                "verbalizer": verbalization.model_name,
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
