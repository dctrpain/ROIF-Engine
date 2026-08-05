"""
ROIF Engine
Kernel Principles

This module defines the immutable architectural principles of the ROIF
Kernel.

These principles describe *what the engine is allowed to become* rather
than *how individual algorithms are implemented*.

Every future subsystem (history, evaluation, planning, cognition,
simulation, learning, etc.) must satisfy these principles.

The kernel itself must never violate them.

Author:
    Architect (Dctr Pain)

License:
    See project license.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple


# ============================================================
# Kernel Principles
# ============================================================


class KernelPrinciple(str, Enum):
    """
    Immutable architectural principles of ROIF.
    """

    EVERYTHING_IS_STRUCTURE = "everything_is_structure"

    EVERYTHING_EVOLVES = "everything_evolves"

    MEMORY_IS_STRUCTURE = "memory_is_structure"

    RECURSIVE_EVOLUTION = "recursive_evolution"

    STRUCTURAL_INHERITANCE = "structural_inheritance"

    INTERFACE_IS_NOT_REALITY = "interface_is_not_reality"

    EVALUATION_PRECEDES_ADAPTATION = (
        "evaluation_precedes_adaptation"
    )

    COUNTERFACTUAL_PRECEDES_INTERVENTION = (
        "counterfactual_precedes_intervention"
    )

    DO_NO_HARM = "do_no_harm"

    EVERY_OUTPUT_BECOMES_NEXT_INPUT = (
        "every_output_becomes_next_input"
    )


# ============================================================
# Immutable Principle
# ============================================================


@dataclass(frozen=True, slots=True)
class PrincipleDefinition:
    """
    Immutable description of one kernel principle.
    """

    principle: KernelPrinciple

    title: str

    description: str

    rationale: str

    immutable: bool = True


# ============================================================
# Kernel Constitution
# ============================================================


KERNEL_CONSTITUTION: Tuple[
    PrincipleDefinition,
    ...
] = (

    PrincipleDefinition(

        principle=KernelPrinciple.EVERYTHING_IS_STRUCTURE,

        title="Everything is Structure",

        description=(
            "Every observable entity is represented as a structure "
            "independent of its physical, biological or conceptual "
            "nature."
        ),

        rationale=(
            "A common structural representation enables unified "
            "analysis across all domains."
        ),
    ),

    PrincipleDefinition(

        principle=KernelPrinciple.EVERYTHING_EVOLVES,

        title="Everything Evolves",

        description=(
            "Every interaction changes the internal or external "
            "structure."
        ),

        rationale=(
            "Static structures are considered a limiting case of "
            "dynamic systems."
        ),
    ),

    PrincipleDefinition(

        principle=KernelPrinciple.MEMORY_IS_STRUCTURE,

        title="Memory is Structural",

        description=(
            "Memory is represented by structural modification rather "
            "than event logging."
        ),

        rationale=(
            "Persistent structural change contains more information "
            "than historical events alone."
        ),
    ),

    PrincipleDefinition(

        principle=KernelPrinciple.RECURSIVE_EVOLUTION,

        title="Recursive Evolution",

        description=(
            "Every completed analysis becomes part of the next state "
            "of the engine."
        ),

        rationale=(
            "The engine continuously evolves through accumulated "
            "experience."
        ),
    ),

    PrincipleDefinition(

        principle=KernelPrinciple.STRUCTURAL_INHERITANCE,

        title="Structural Inheritance",

        description=(
            "New structures may inherit validated structural priors "
            "without inheriting final conclusions."
        ),

        rationale=(
            "Evolution preserves useful organization while allowing "
            "independent adaptation."
        ),
    ),

    PrincipleDefinition(

        principle=KernelPrinciple.INTERFACE_IS_NOT_REALITY,

        title="Interface is not Reality",

        description=(
            "Sensors and interfaces provide observations, not absolute "
            "truth."
        ),

        rationale=(
            "All observations are interpreted through structural "
            "models."
        ),
    ),

    PrincipleDefinition(

        principle=KernelPrinciple.EVALUATION_PRECEDES_ADAPTATION,

        title="Evaluation Precedes Adaptation",

        description=(
            "Every adaptive modification requires structural "
            "evaluation."
        ),

        rationale=(
            "Uncontrolled adaptation may reduce global integrity."
        ),
    ),

    PrincipleDefinition(

        principle=KernelPrinciple.COUNTERFACTUAL_PRECEDES_INTERVENTION,

        title="Counterfactual Before Intervention",

        description=(
            "Possible future outcomes should be evaluated before "
            "changing the observed system."
        ),

        rationale=(
            "Prediction reduces unnecessary structural damage."
        ),
    ),

    PrincipleDefinition(

        principle=KernelPrinciple.DO_NO_HARM,

        title="Do No Harm",

        description=(
            "The engine shall preserve or improve long-term structural "
            "integrity whenever possible."
        ),

        rationale=(
            "Local optimization must never knowingly reduce global "
            "structural coherence."
        ),
    ),

    PrincipleDefinition(

        principle=KernelPrinciple.EVERY_OUTPUT_BECOMES_NEXT_INPUT,

        title="Recursive Feedback",

        description=(
            "Every accepted result becomes part of the future "
            "knowledge state."
        ),

        rationale=(
            "Recursive accumulation enables continual evolution."
        ),
    ),
)


# ============================================================
# Public API
# ============================================================


def principles() -> Tuple[PrincipleDefinition, ...]:
    """
    Return the immutable ROIF kernel constitution.
    """

    return KERNEL_CONSTITUTION


def principle(
    key: KernelPrinciple,
) -> PrincipleDefinition:
    """
    Return one immutable kernel principle.
    """

    for item in KERNEL_CONSTITUTION:

        if item.principle == key:
            return item

    raise KeyError(key)