from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import re

from .expression_contract import (
    ExpressionClaim,
    ExpressionClaimKind,
    ExpressionContractBuilder,
)
from .expression_gate import ExpressionDecision


class ValidationStatus(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class ExpressionValidationResult:
    """
    Result of validating generated language against an already
    admitted and contract-bound RDA expression.

    Validation does not determine RDA internal state.

    It protects against:
        - unsupported added content,
        - unsupported numeric claims,
        - loss of required admitted claims.
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
    Conservative outward-expression validator.

    Authoritative chain:

        RDA evidence
            -> ExpressionDecision
            -> ExpressionContract
            -> canonical expression

    Generated language is subordinate to this chain.

    The validator must never create, repair, strengthen, or
    reinterpret RDA evidence.
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

    def __init__(self) -> None:
        self._contract_builder = ExpressionContractBuilder()

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

        # -----------------------------------------------------
        # 1. Unsupported semantic additions.
        # -----------------------------------------------------

        for term in self._FORBIDDEN_SEMANTIC_TERMS:
            if term.casefold() in candidate_lower:
                reasons.append(
                    f"unsupported_semantic_term:{term}"
                )

        # -----------------------------------------------------
        # 2. Unsupported numeric additions.
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # 3. Semantic-preservation contract.
        # -----------------------------------------------------

        contract = self._contract_builder.build(decision)

        for claim in contract.claims:
            if not self._claim_preserved(
                claim=claim,
                candidate_lower=candidate_lower,
                candidate_numbers=candidate_numbers,
            ):
                if self._is_numeric_claim(claim.kind):
                    reasons.append(
                        "missing_required_numeric_claim:"
                        f"{claim.kind.value}"
                    )
                else:
                    reasons.append(
                        "missing_required_claim:"
                        f"{claim.kind.value}"
                    )

        return self._result(
            canonical=canonical,
            candidate=candidate,
            reasons=self._deduplicate(reasons),
        )

    def _claim_preserved(
        self,
        *,
        claim: ExpressionClaim,
        candidate_lower: str,
        candidate_numbers: tuple[float, ...],
    ) -> bool:
        kind = claim.kind

        if self._is_numeric_claim(kind):
            return self._numeric_claim_preserved(
                claim=claim,
                candidate_numbers=candidate_numbers,
                candidate_lower=candidate_lower,
            )

        if kind is ExpressionClaimKind.STATE_CHANGE_PRESENT:
            return self._contains_any(
                candidate_lower,
                (
                    "измен",
                    "отлич",
                    "отклон",
                    "не совпад",
                    "выш",
                ),
            )

        if (
            kind
            is ExpressionClaimKind.DIFFERENCE_STRUCTURE_PRESENT
        ):
            return self._contains_any(
                candidate_lower,
                (
                    "различ",
                    "отклон",
                    "несовпад",
                    "разниц",
                    "структур",
                    "профил",
                ),
            )

        if (
            kind
            is ExpressionClaimKind.RELATIONAL_STRUCTURE_PRESENT
        ):
            return self._contains_any(
                candidate_lower,
                (
                    "совместн",
                    "связ",
                    "отношен",
                    "структур",
                    "групп",
                ),
            )

        if (
            kind
            is ExpressionClaimKind
            .REPRESENTATIONAL_INSUFFICIENCY_PRESENT
        ):
            return self._contains_any(
                candidate_lower,
                (
                    "недостаточ",
                    "не хватает",
                    "не охватыва",
                    "не описыва",
                    "не позволяет полностью",
                ),
            )

        if (
            kind
            is ExpressionClaimKind
            .REPEATING_EXPERIENCE_STRUCTURE_PRESENT
        ):
            has_experience = self._contains_any(
                candidate_lower,
                (
                    "опыт",
                    "пережива",
                ),
            )

            has_repetition = self._contains_any(
                candidate_lower,
                (
                    "повторя",
                    "повторн",
                    "повторяем",
                    "устойчив",
                    "регуляр",
                ),
            )

            has_structure = self._contains_any(
                candidate_lower,
                (
                    "структур",
                    "закономер",
                    "паттерн",
                    "отношен",
                    "организац",
                ),
            )

            return (
                has_experience
                and has_repetition
                and has_structure
            )

        raise ValueError(
            f"Unsupported ExpressionClaimKind: {kind!r}"
        )

    @staticmethod
    def _is_numeric_claim(
        kind: ExpressionClaimKind,
    ) -> bool:
        return kind in {
            ExpressionClaimKind.GROUP_COUNT,
            ExpressionClaimKind.STRONGEST_GROUP_STRENGTH,
            ExpressionClaimKind.REPRESENTED_DIMENSION,
            ExpressionClaimKind
            .PERSISTENT_RESIDUAL_DIMENSION_COUNT,
            ExpressionClaimKind.RESIDUAL_VARIANCE_FRACTION,
            ExpressionClaimKind.DEVIATION_SCORE,
            ExpressionClaimKind.MAXIMUM_ABSOLUTE_DEVIATION,
            ExpressionClaimKind.L2_DEVIATION_NORM,
        }

    _RUSSIAN_INTEGER_FORMS = {
        0: ("ноль", "нулевой", "нулевая", "нулевое"),
        1: (
            "один", "одна", "одно",
            "одного", "одной",
            "одним", "одною",
        ),
        2: (
            "два", "две",
            "двух",
            "двум", "двумя",
        ),
        3: (
            "три",
            "трёх", "трех",
            "трём", "трем",
            "тремя",
        ),
        4: (
            "четыре",
            "четырёх", "четырех",
            "четырём", "четырем",
            "четырьмя",
        ),
        5: (
            "пять",
            "пяти",
            "пятью",
        ),
    }

    @classmethod
    def _numeric_claim_preserved(
        cls,
        *,
        claim: ExpressionClaim,
        candidate_numbers: tuple[float, ...],
        candidate_lower: str = "",
    ) -> bool:
        expected = float(claim.value)

        allowed = (
            expected,
            round(expected, 4),
        )

        if any(
            math.isclose(
                candidate,
                allowed_value,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            for candidate in candidate_numbers
            for allowed_value in allowed
        ):
            return True

        if expected.is_integer():
            integer_value = int(expected)

            forms = cls._RUSSIAN_INTEGER_FORMS.get(
                integer_value,
                (),
            )

            words = set(
                re.findall(
                    r"[а-яё]+",
                    candidate_lower,
                )
            )

            if any(
                form in words
                for form in forms
            ):
                return True

        return False

    @staticmethod
    def _contains_any(
        text: str,
        fragments: tuple[str, ...],
    ) -> bool:
        return any(
            fragment in text
            for fragment in fragments
        )

    @staticmethod
    def _extract_numbers(
        text: str,
    ) -> tuple[float, ...]:
        matches = re.findall(
            r"(?<![\w])[-+]?(?:\d+(?:[.,]\d+)?|\.[0-9]+)(?:[eE][-+]?\d+)?",
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
    def _deduplicate(
        reasons: list[str],
    ) -> list[str]:
        return list(dict.fromkeys(reasons))

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

