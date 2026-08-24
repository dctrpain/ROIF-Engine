from __future__ import annotations

import json

from experiments.roif_rda_3_first_dimensional_growth import (
    run_first_dimensional_growth,
)
from roif.development.expression_gate import ExpressionGate
from roif.development.language_adapter import LanguageAdapter
from roif.development.llm_verbalizer import LLMVerbalizer
from roif.development.ollama_backend import OllamaBackend


def main() -> None:
    result, assessment = run_first_dimensional_growth(
        return_internal=True,
    )

    gate = ExpressionGate()

    decision = gate.decide(
        dimensional_growth=assessment,
    )

    canonical_adapter = LanguageAdapter()
    canonical = canonical_adapter.render(decision)

    verbalizer = LLMVerbalizer(
        backend=OllamaBackend(
            model="gemma3:4b",
        )
    )

    natural = verbalizer.speak(
        decision,
        language="ru",
    )

    print()
    print("=== RDA CANONICAL EXPRESSION ===")
    print()
    print(canonical.text or "[no expression]")

    print()
    print("=== RDA LLM VERBALIZATION ===")
    print()
    print(natural.text or "[no expression]")

    print()
    print("=== EXPRESSION KIND ===")
    print(decision.kind.value)

    print()
    print("=== TRACE ===")
    print(
        json.dumps(
            canonical.trace,
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
                "verbalizer": natural.model_name,
                "used_llm": natural.used_llm,
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
