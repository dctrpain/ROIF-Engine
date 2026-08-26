from __future__ import annotations

from dataclasses import dataclass, field

from .dimensional_growth import DimensionalGrowthAssessment
from .endogenous_difference import DifferenceProfile
from .endogenous_grouping import GroupingResult
from .familiar_state_change import ChangeDetection
from .observation import Observation


@dataclass(slots=True)
class DevelopmentalState:
    """
    Persistent endogenous developmental state.

    This object stores developmental experience and internally produced
    structures across time.

    Fundamental invariant:

        DevelopmentalState is memory, not a developmental script.

    In particular, it contains no:
        - RDA stage counter,
        - expected next stage,
        - transition table,
        - externally supplied developmental target,
        - reward or goal,
        - semantic event label,
        - world-ground-truth transition instruction.

    Developmental mechanisms may inspect accumulated state and add new
    internally supported structures when their own evidential criteria
    are satisfied.
    """

    observations: list[Observation] = field(
        default_factory=list
    )

    change_detections: list[ChangeDetection] = field(
        default_factory=list
    )

    difference_profiles: list[DifferenceProfile] = field(
        default_factory=list
    )

    grouping_results: list[GroupingResult] = field(
        default_factory=list
    )

    dimensional_growth_assessments: list[
        DimensionalGrowthAssessment
    ] = field(
        default_factory=list
    )

    def record_observation(
        self,
        observation: Observation,
    ) -> None:
        self.observations.append(observation)

    def record_change_detection(
        self,
        detection: ChangeDetection,
    ) -> None:
        self.change_detections.append(detection)

    def record_difference_profile(
        self,
        profile: DifferenceProfile,
    ) -> None:
        self.difference_profiles.append(profile)

    def record_grouping_result(
        self,
        grouping: GroupingResult,
    ) -> None:
        self.grouping_results.append(grouping)

    def record_dimensional_growth_assessment(
        self,
        assessment: DimensionalGrowthAssessment,
    ) -> None:
        self.dimensional_growth_assessments.append(
            assessment
        )

    @property
    def observation_count(self) -> int:
        return len(self.observations)

    @property
    def change_detection_count(self) -> int:
        return len(self.change_detections)

    @property
    def difference_profile_count(self) -> int:
        return len(self.difference_profiles)

    @property
    def grouping_result_count(self) -> int:
        return len(self.grouping_results)

    @property
    def dimensional_growth_assessment_count(self) -> int:
        return len(
            self.dimensional_growth_assessments
        )

    @property
    def latest_change_detection(
        self,
    ) -> ChangeDetection | None:
        if not self.change_detections:
            return None

        return self.change_detections[-1]

    @property
    def latest_difference_profile(
        self,
    ) -> DifferenceProfile | None:
        if not self.difference_profiles:
            return None

        return self.difference_profiles[-1]

    @property
    def latest_grouping_result(
        self,
    ) -> GroupingResult | None:
        if not self.grouping_results:
            return None

        return self.grouping_results[-1]

    @property
    def latest_dimensional_growth_assessment(
        self,
    ) -> DimensionalGrowthAssessment | None:
        if not self.dimensional_growth_assessments:
            return None

        return self.dimensional_growth_assessments[-1]
