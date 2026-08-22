from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from roif.development.endogenous_difference import (
    DifferenceProfile,
    EndogenousDifferenceAnalyzer,
)
from roif.development.endogenous_grouping import (
    ChannelRelation,
    EndogenousGroup,
    EndogenousGroupingAnalyzer,
    GroupingResult,
)
from roif.development.experience_stream import ExperienceStream
from roif.worlds.tensegrity_3d_world import Tensegrity3DWorld


BENCHMARK_VERSION = "roif_rda_2_first_endogenous_grouping_v1"

FROZEN_DIFFERENCE_ANALYZER_COMMIT = "0aee759"
FROZEN_GROUPING_ANALYZER_COMMIT = "88ab5cf"

FROZEN_DIFFERENCE_SCALE_FLOOR = 1e-9
FROZEN_RELATION_THRESHOLD = 0.95
FROZEN_MINIMUM_TEMPORAL_VARIATION = 1e-12


@dataclass(frozen=True, slots=True)
class FirstEndogenousGroupingConfig:
    """
    First real exposure of the frozen endogenous grouping analyzer.

    The grouping analyzer receives only a temporal sequence of
    DifferenceProfile objects derived from embodied Observation objects.

    World topology and perturbation metadata remain experimenter-side
    ground truth only.
    """

    duration: float = 5.0
    dt: float = 0.001
    baseline_duration: float = 1.0
    perturbation_time: float = 2.0
    grouping_window_duration: float = 1.0

    perturbation_node: int = 0
    perturbation_force_x: float = 1.0
    perturbation_force_y: float = 0.0
    perturbation_force_z: float = 0.0

    def __post_init__(self) -> None:
        numeric_values = (
            self.duration,
            self.dt,
            self.baseline_duration,
            self.perturbation_time,
            self.grouping_window_duration,
            self.perturbation_force_x,
            self.perturbation_force_y,
            self.perturbation_force_z,
        )

        if not all(
            np.isfinite(value)
            for value in numeric_values
        ):
            raise ValueError(
                "All numeric configuration values must be finite."
            )

        if self.duration <= 0.0:
            raise ValueError(
                "duration must be > 0."
            )

        if self.dt <= 0.0:
            raise ValueError(
                "dt must be > 0."
            )

        if not (
            0.0
            < self.baseline_duration
            < self.perturbation_time
            < self.duration
        ):
            raise ValueError(
                "Required ordering: "
                "0 < baseline_duration < perturbation_time < duration."
            )

        if self.grouping_window_duration <= 0.0:
            raise ValueError(
                "grouping_window_duration must be > 0."
            )

        if (
            self.perturbation_time
            + self.grouping_window_duration
            > self.duration
        ):
            raise ValueError(
                "Grouping window must fit inside the run."
            )

        if self.perturbation_node < 0:
            raise ValueError(
                "perturbation_node must be >= 0."
            )


def _canonical_json_bytes(
    value: Any,
) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(
    value: Any,
) -> str:
    return hashlib.sha256(
        _canonical_json_bytes(value)
    ).hexdigest()


def _integer_step(
    *,
    time_value: float,
    dt: float,
    name: str,
) -> int:
    step = int(
        round(
            time_value / dt
        )
    )

    if not np.isclose(
        step * dt,
        time_value,
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError(
            f"{name} must be an integer multiple of dt."
        )

    return step


def _relation_to_record(
    relation: ChannelRelation,
) -> dict[str, Any]:
    return {
        "channel_a": relation.channel_a,
        "channel_b": relation.channel_b,
        "similarity": relation.similarity,
        "absolute_similarity": (
            relation.absolute_similarity
        ),
        "sample_count": relation.sample_count,
    }


def _group_to_record(
    group: EndogenousGroup,
) -> dict[str, Any]:
    return {
        "group_id": group.group_id,
        "channel_names": list(
            group.channel_names
        ),
        "channel_count": len(
            group.channel_names
        ),
        "internal_relation_strength": (
            group.internal_relation_strength
        ),
    }


def _result_to_records(
    grouping: GroupingResult,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    relations = [
        _relation_to_record(
            relation
        )
        for relation in grouping.relations
    ]

    groups = [
        _group_to_record(
            group
        )
        for group in grouping.groups
    ]

    return relations, groups


def _profile_from_observation(
    *,
    analyzer: EndogenousDifferenceAnalyzer,
    observation,
) -> DifferenceProfile:
    return analyzer.analyze(
        observation
    )


def run_first_endogenous_grouping(
    config: FirstEndogenousGroupingConfig | None = None,
) -> dict[str, Any]:
    """
    First blind real exposure of the frozen endogenous grouping analyzer.

    Architecture:

        familiar Observation history
            ->
        frozen endogenous difference analyzer
            ->
        DifferenceProfile trajectory
            ->
        frozen endogenous grouping analyzer
            ->
        relations + endogenous groups

    Neither analyzer receives world topology or perturbation metadata.
    """

    if config is None:
        config = FirstEndogenousGroupingConfig()

    total_steps = _integer_step(
        time_value=config.duration,
        dt=config.dt,
        name="duration",
    )

    baseline_steps = _integer_step(
        time_value=config.baseline_duration,
        dt=config.dt,
        name="baseline_duration",
    )

    perturbation_step = _integer_step(
        time_value=config.perturbation_time,
        dt=config.dt,
        name="perturbation_time",
    )

    grouping_window_steps = _integer_step(
        time_value=config.grouping_window_duration,
        dt=config.dt,
        name="grouping_window_duration",
    )

    grouping_window_end_step = (
        perturbation_step
        + grouping_window_steps
    )

    world = Tensegrity3DWorld()

    if config.perturbation_node >= world.node_count:
        raise ValueError(
            "perturbation_node is outside the body."
        )

    body = world.body(
        "tensegrity_body_0"
    )

    experience = ExperienceStream(
        body_id=body.body_id
    )

    difference_analyzer = (
        EndogenousDifferenceAnalyzer(
            absolute_scale_floor=(
                FROZEN_DIFFERENCE_SCALE_FLOOR
            )
        )
    )

    grouping_analyzer = (
        EndogenousGroupingAnalyzer(
            relation_threshold=(
                FROZEN_RELATION_THRESHOLD
            ),
            minimum_temporal_variation=(
                FROZEN_MINIMUM_TEMPORAL_VARIATION
            ),
        )
    )

    # ---------------------------------------------------------
    # Phase 1: familiar baseline.
    # ---------------------------------------------------------

    first_observation = body.observe()

    experience.append(
        first_observation
    )

    baseline_observations = [
        first_observation
    ]

    for _step_index in range(
        1,
        baseline_steps + 1,
    ):
        world.step(
            config.dt
        )

        observation = body.observe()

        experience.append(
            observation
        )

        baseline_observations.append(
            observation
        )

    difference_analyzer.fit(
        baseline_observations
    )

    # ---------------------------------------------------------
    # Phase 2: produce endogenous DifferenceProfile trajectory.
    # ---------------------------------------------------------

    all_profiles: list[
        DifferenceProfile
    ] = []

    grouping_profiles: list[
        DifferenceProfile
    ] = []

    grouping_profile_step_indices: list[
        int
    ] = []

    for step_index in range(
        baseline_steps + 1,
        total_steps + 1,
    ):
        if step_index == perturbation_step:
            world.apply_external_force(
                node_index=(
                    config.perturbation_node
                ),
                force=np.asarray(
                    [
                        config.perturbation_force_x,
                        config.perturbation_force_y,
                        config.perturbation_force_z,
                    ],
                    dtype=float,
                ),
            )

        world.step(
            config.dt
        )

        observation = body.observe()

        experience.append(
            observation
        )

        profile = _profile_from_observation(
            analyzer=difference_analyzer,
            observation=observation,
        )

        all_profiles.append(
            profile
        )

        if (
            perturbation_step
            <= step_index
            <= grouping_window_end_step
        ):
            grouping_profiles.append(
                profile
            )

            grouping_profile_step_indices.append(
                step_index
            )

    if len(grouping_profiles) < 2:
        raise RuntimeError(
            "Grouping window produced fewer than two profiles."
        )

    # ---------------------------------------------------------
    # Phase 3: frozen grouping analyzer sees only profile history.
    # ---------------------------------------------------------

    grouping = grouping_analyzer.analyze(
        grouping_profiles
    )

    relation_records, group_records = (
        _result_to_records(
            grouping
        )
    )

    # ---------------------------------------------------------
    # Post-hoc summaries.
    # ---------------------------------------------------------

    relation_records_sorted = sorted(
        relation_records,
        key=lambda item: (
            -item["absolute_similarity"],
            item["channel_a"],
            item["channel_b"],
        ),
    )

    strongest_positive_relations = [
        item
        for item in relation_records_sorted
        if item["similarity"] > 0.0
    ][:10]

    strongest_negative_relations = [
        item
        for item in relation_records_sorted
        if item["similarity"] < 0.0
    ][:10]

    grouped_channels = {
        channel_name
        for group in grouping.groups
        for channel_name in group.channel_names
    }

    ungrouped_channels = [
        channel_name
        for channel_name in grouping.channel_names
        if channel_name not in grouped_channels
    ]

    group_sizes = [
        len(
            group.channel_names
        )
        for group in grouping.groups
    ]

    largest_group_size = (
        max(
            group_sizes
        )
        if group_sizes
        else 0
    )

    mean_group_size = (
        float(
            np.mean(
                group_sizes
            )
        )
        if group_sizes
        else 0.0
    )

    result = {
        "benchmark": BENCHMARK_VERSION,
        "developmental_stage": "RDA-2",
        "experience": "first_endogenous_grouping",
        "frozen_components": {
            "difference_analyzer_commit": (
                FROZEN_DIFFERENCE_ANALYZER_COMMIT
            ),
            "grouping_analyzer_commit": (
                FROZEN_GROUPING_ANALYZER_COMMIT
            ),
            "difference_scale_floor": (
                FROZEN_DIFFERENCE_SCALE_FLOOR
            ),
            "relation_threshold": (
                FROZEN_RELATION_THRESHOLD
            ),
            "minimum_temporal_variation": (
                FROZEN_MINIMUM_TEMPORAL_VARIATION
            ),
            "parameters_modified_after_freeze": False,
        },
        "epistemic_boundary": {
            "grouping_received_ground_truth": False,
            "grouping_received_world_state": False,
            "grouping_received_body_ground_truth": False,
            "grouping_received_topology": False,
            "grouping_received_node_mapping": False,
            "grouping_received_member_mapping": False,
            "grouping_received_perturbation_time": False,
            "grouping_received_perturbation_node": False,
            "grouping_received_force_vector": False,
            "grouping_received_world_coordinates": False,
            "grouping_received_body_part_labels": False,
            "grouping_received_reward": False,
            "grouping_received_goal": False,
            "grouping_received_event_label": False,
            "grouping_received_valence": False,
        },
        "configuration": {
            "duration": config.duration,
            "dt": config.dt,
            "baseline_duration": (
                config.baseline_duration
            ),
            "baseline_sample_count": (
                len(
                    baseline_observations
                )
            ),
            "grouping_window_duration": (
                config.grouping_window_duration
            ),
            "grouping_profile_count": (
                len(
                    grouping_profiles
                )
            ),
            "all_profile_count": (
                len(
                    all_profiles
                )
            ),
        },
        "ground_truth_evaluation_only": {
            "perturbation_time": (
                config.perturbation_time
            ),
            "perturbation_step": (
                perturbation_step
            ),
            "perturbation_node": (
                config.perturbation_node
            ),
            "perturbation_force": [
                config.perturbation_force_x,
                config.perturbation_force_y,
                config.perturbation_force_z,
            ],
            "grouping_window_start_step": (
                perturbation_step
            ),
            "grouping_window_end_step": (
                grouping_window_end_step
            ),
        },
        "grouping_summary": {
            "channel_count": (
                len(
                    grouping.channel_names
                )
            ),
            "relation_count": (
                len(
                    grouping.relations
                )
            ),
            "group_count": (
                len(
                    grouping.groups
                )
            ),
            "group_sizes": (
                group_sizes
            ),
            "largest_group_size": (
                largest_group_size
            ),
            "mean_group_size": (
                mean_group_size
            ),
            "grouped_channel_count": (
                len(
                    grouped_channels
                )
            ),
            "ungrouped_channel_count": (
                len(
                    ungrouped_channels
                )
            ),
            "ungrouped_channels": (
                ungrouped_channels
            ),
            "strongest_positive_relations": (
                strongest_positive_relations
            ),
            "strongest_negative_relations": (
                strongest_negative_relations
            ),
        },
        "groups": (
            group_records
        ),
        "relations": (
            relation_records
        ),
        "grouping_profile_step_indices": (
            grouping_profile_step_indices
        ),
    }

    result["relations_sha256"] = _sha256(
        relation_records
    )

    result["groups_sha256"] = _sha256(
        group_records
    )

    result["grouping_window_sha256"] = _sha256(
        {
            "step_indices": (
                grouping_profile_step_indices
            ),
            "profile_sequence_indices": [
                profile.sequence_index
                for profile in grouping_profiles
            ],
            "profile_timestamps": [
                profile.timestamp
                for profile in grouping_profiles
            ],
        }
    )

    result["run_sha256"] = _sha256(
        {
            key: value
            for key, value in result.items()
            if key != "run_sha256"
        }
    )

    return result


def save_result(
    result: dict[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run first blind exposure of the frozen "
            "ROIF endogenous grouping analyzer."
        )
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmark_results/"
            "rda_2_first_endogenous_grouping_v1.json"
        ),
    )

    args = parser.parse_args()

    result = run_first_endogenous_grouping()

    save_result(
        result=result,
        output_path=args.output,
    )

    summary = {
        "benchmark": (
            result["benchmark"]
        ),
        "frozen_grouping_commit": (
            result["frozen_components"][
                "grouping_analyzer_commit"
            ]
        ),
        "grouping_summary": (
            result["grouping_summary"]
        ),
        "groups": (
            result["groups"]
        ),
        "relations_sha256": (
            result["relations_sha256"]
        ),
        "groups_sha256": (
            result["groups_sha256"]
        ),
        "grouping_window_sha256": (
            result["grouping_window_sha256"]
        ),
        "run_sha256": (
            result["run_sha256"]
        ),
    }

    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
