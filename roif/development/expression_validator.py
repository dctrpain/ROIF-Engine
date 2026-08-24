from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import re

from .expression_gate import ExpressionDecision


class ValidationStatus(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class ExpressionValidationResult:
    """
    Result of validating a generated linguistic candidate against
    an already admitted RDA expression.

    Validation does not determine internal state or meaning.
    It only protects the boundary between supported evidence and
    outward linguistic expression.
    """

    status: ValidationStatus
    output_text: str
    candidate_text: str
    canonical_text: str
    reasons: tuple[str, ...]

    @property
    def accepted(self) -> bool:
        return self.status is ValidationStatus.ACCEPTED


class ExpressionValidator:
    """
    Conservative validator for optional generative verbalization.

    Fundamental invariant:

        RDA evidence -> ExpressionDecision -> canonical expression

    is authoritative.

    Generated language is subordinate to that chain.

    The validator may reject generated language, but it must never
    create, strengthen, reinterpret, or repair RDA evidence.
    """

    _FORBIDDEN_SEMANTIC_TERMS = (
        "я чувствую",
        "я ощущаю",
        "я хочу",
        "я думаю",
        "я считаю",
        "я понимаю",
        "я знаю",
        "мне кажется",
        "моя цель",
        "моё намерение",
        "мое намерение",
        "моя эмоция",
        "мои эмоции",
        "моя потребность",
        "мои потребности",
        "моя внутренняя модель",
    )

    def validate(
        self,
        *,
        decision: ExpressionDecision,
        canonical_text: str,
        candidate_text: str,
    ) -> ExpressionValidationResult:
        canonical = canonical_text.strip()
        candidate = candidate_text.strip()

        reasons: list[str] = []

        if not canonical:
            if candidate:
                reasons.append(
                    "candidate_exists_without_canonical_expression"
                )

            return self._result(
                canonical=canonical,
                candidate=candidate,
                reasons=reasons,
            )

        if not candidate:
            reasons.append("empty_candidate")

            return self._result(
                canonical=canonical,
                candidate=candidate,
                reasons=reasons,
            )

        candidate_lower = candidate.casefold()

        for term in self._FORBIDDEN_SEMANTIC_TERMS:
            if term.casefold() in candidate_lower:
                reasons.append(
                    f"unsupported_semantic_term:{term}"
                )

        expected_numbers = self._supported_numeric_tokens(
            decision
        )

        candidate_numbers = self._extract_numbers(candidate)

        for number in candidate_numbers:
            if not self._matches_supported_number(
                number,
                expected_numbers,
            ):
                reasons.append(
                    f"unsupported_numeric_claim:{number}"
                )

        return self._result(
            canonical=canonical,
            candidate=candidate,
            reasons=reasons,
        )

    @staticmethod
    def _extract_numbers(text: str) -> tuple[float, ...]:
        matches = re.findall(
            r"(?<![\w])[-+]?\d+(?:[.,]\d+)?",
            text,
        )

        values: list[float] = []

        for match in matches:
            try:
                values.append(
                    float(match.replace(",", "."))
                )
            except ValueError:
                continue

        return tuple(values)

    @staticmethod
    def _supported_numeric_tokens(
        decision: ExpressionDecision,
    ) -> tuple[float, ...]:
        evidence = decision.evidence

        raw_values = (
            evidence.sequence_index,
            evidence.timestamp,
            evidence.deviation_score,
            evidence.maximum_absolute_deviation,
            evidence.l2_deviation_norm,
            evidence.group_count,
            evidence.strongest_group_strength,
            evidence.represented_dimension,
            evidence.residual_variance_fraction,
            evidence.persistent_residual_dimension_count,
        )

        values: list[float] = []

        for value in raw_values:
            if value is None:
                continue

            if isinstance(value, bool):
                continue

            numeric = float(value)

            if math.isfinite(numeric):
                values.append(numeric)

                # Canonical adapters may deliberately round evidence
                # for human-readable expression.
                values.append(round(numeric, 4))

        return tuple(values)

    @staticmethod
    def _matches_supported_number(
        candidate: float,
        supported: tuple[float, ...],
    ) -> bool:
        return any(
            math.isclose(
                candidate,
                expected,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            for expected in supported
        )

    @staticmethod
    def _result(
        *,
        canonical: str,
        candidate: str,
        reasons: list[str],
    ) -> ExpressionValidationResult:
        if reasons:
            return ExpressionValidationResult(
                status=ValidationStatus.REJECTED,
                output_text=canonical,
                candidate_text=candidate,
                canonical_text=canonical,
                reasons=tuple(reasons),
            )

        return ExpressionValidationResult(
            status=ValidationStatus.ACCEPTED,
            output_text=candidate,
            candidate_text=candidate,
            canonical_text=canonical,
            reasons=(),
        )
