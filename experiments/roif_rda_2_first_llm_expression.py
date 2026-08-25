from __future__ import annotations

import json

from experiments.roif_rda_2_first_endogenous_grouping import (
    run_first_endogenous_grouping,
)
from roif.development.expression_gate import ExpressionGate
from roif.development.language_adapter import LanguageAdapter
from roif.development.llm_verbalizer import LLMVerbalizer
from roif.development.ollama_backend import OllamaBackend


def main() -> None:
    result, grouping = run_first_endogenous_grouping(
        return_internal=True,
    )

    gate = ExpressionGate()

    decision = gate.decide(
        grouping=grouping,
    )

    adapter = LanguageAdapter()

    canonical = adapter.render(
        decision
    )

    verbalizer = LLMVerbalizer(
        backend=OllamaBackend(),
    )

    verbalization = verbalizer.speak(
        decision,
        language="ru",
    )

    print()
    print("=== RDA-2 FIRST ENDOGENOUS GROUPING ===")

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

    strongest_group_strength = (
        None
        if not grouping.groups
        else max(
            group.internal_relation_strength
            for group in grouping.groups
        )
    )

    internal_grouping = {
        "channel_count": len(
            grouping.channel_names
        ),
        "relation_count": len(
            grouping.relations
        ),
        "group_count": len(
            grouping.groups
        ),
        "strongest_group_strength": (
            strongest_group_strength
        ),
    }

    print()
    print("=== INTERNAL GROUPING RESULT ===")
    print(
        json.dumps(
            internal_grouping,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )

    source = {
        "benchmark": result["benchmark"],
        "developmental_stage": "RDA-2",
        "experience": "first_endogenous_grouping",
        "run_sha256": result["run_sha256"],
        "used_llm": verbalization.used_llm,
        "validation_status": (
            verbalization.validation_status
        ),
        "verbalizer": (
            verbalization.model_name
        ),
    }

    print()
    print("=== SOURCE ===")
    print(
        json.dumps(
            source,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
