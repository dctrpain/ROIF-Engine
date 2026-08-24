from __future__ import annotations

import json

from experiments.roif_rda_3_first_dimensional_growth import (
    run_first_dimensional_growth,
)
from roif.development.expression_gate import ExpressionGate
from roif.development.language_adapter import LanguageAdapter


def main() -> None:
    result, assessment = run_first_dimensional_growth(
        return_internal=True,
    )

    gate = ExpressionGate()

    decision = gate.decide(
        dimensional_growth=assessment,
    )

    adapter = LanguageAdapter()

    expression = adapter.render(
        decision,
    )

    print()
    print("=== RDA FIRST EXPRESSION ===")
    print()

    if expression.text:
        print(expression.text)
    else:
        print("[no expression]")

    print()
    print("=== EXPRESSION TRACE ===")
    print(
        json.dumps(
            expression.trace,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )

    print()
    print("=== SOURCE RUN ===")
    print(
        json.dumps(
            {
                "benchmark": result["benchmark"],
                "developmental_stage": result[
                    "developmental_stage"
                ],
                "experience": result["experience"],
                "run_sha256": result["run_sha256"],
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
