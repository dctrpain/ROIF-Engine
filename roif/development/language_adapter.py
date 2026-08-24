from __future__ import annotations

from dataclasses import dataclass

from .expression_gate import (
    ExpressionDecision,
    ExpressionKind,
)


@dataclass(frozen=True, slots=True)
class LanguageExpression:
    """
    Human-readable rendering of an already formed ExpressionDecision.

    Fundamental invariant:

        LanguageExpression does not create internal content.

    It may only verbalize information already present in
    ExpressionDecision and its evidence.
    """

    kind: ExpressionKind
    text: str
    trace: dict[str, object]


class LanguageAdapter:
    """
    Deterministic language adapter for the first RDA expression layer.

    No LLM is used here yet.

    This stage exists to establish a strict, testable boundary:

        RDA state
            -> ExpressionDecision
            -> deterministic language
    """

    def render(
        self,
        decision: ExpressionDecision,
    ) -> LanguageExpression:

        evidence = decision.evidence

        trace = {
            "kind": decision.kind.value,
            "sequence_index": evidence.sequence_index,
            "timestamp": evidence.timestamp,
            "deviation_score": evidence.deviation_score,
            "maximum_absolute_deviation": (
                evidence.maximum_absolute_deviation
            ),
            "l2_deviation_norm": evidence.l2_deviation_norm,
            "group_count": evidence.group_count,
            "strongest_group_strength": (
                evidence.strongest_group_strength
            ),
            "represented_dimension": (
                evidence.represented_dimension
            ),
            "residual_variance_fraction": (
                evidence.residual_variance_fraction
            ),
            "persistent_residual_dimension_count": (
                evidence.persistent_residual_dimension_count
            ),
            "growth_supported": evidence.growth_supported,
        }

        if decision.kind is ExpressionKind.NO_EXPRESSION:
            text = ""

        elif decision.kind is ExpressionKind.STATE_CHANGE:
            text = (
                "Моё текущее внутреннее состояние отличается "
                "от ранее сформированной знакомой регулярности."
            )

            if evidence.deviation_score is not None:
                text += (
                    " Величина внутреннего отклонения: "
                    f"{evidence.deviation_score:.6g}."
                )

        elif decision.kind is ExpressionKind.RELATIONAL_STRUCTURE:
            text = (
                "В изменениях моих внутренних каналов обнаружилась "
                "устойчивая совместная структура."
            )

            if evidence.group_count:
                text += (
                    " Число обнаруженных внутренних групп: "
                    f"{evidence.group_count}."
                )

            if evidence.strongest_group_strength is not None:
                text += (
                    " Наибольшая внутренняя связность группы: "
                    f"{evidence.strongest_group_strength:.6g}."
                )

        elif (
            decision.kind
            is ExpressionKind.REPRESENTATIONAL_INSUFFICIENCY
        ):
            text = (
                "Моего текущего внутреннего представления недостаточно "
                "для полного описания повторяющейся структуры моего опыта."
            )

            if evidence.represented_dimension is not None:
                text += (
                    " Сейчас представленная размерность: "
                    f"{evidence.represented_dimension}."
                )

            if (
                evidence.persistent_residual_dimension_count
                is not None
            ):
                text += (
                    " Устойчивая остаточная структура поддерживает "
                    f"{evidence.persistent_residual_dimension_count} "
                    "дополнительных направлений."
                )

            if evidence.residual_variance_fraction is not None:
                text += (
                    " Доля необъяснённой остаточной вариативности: "
                    f"{evidence.residual_variance_fraction:.6g}."
                )

        else:
            raise ValueError(
                f"Unsupported ExpressionKind: {decision.kind!r}"
            )

        return LanguageExpression(
            kind=decision.kind,
            text=text,
            trace=trace,
        )