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
from roif.development.experience_stream import ExperienceStream
from roif.worlds.tensegrity_3d_world import Tensegrity3DWorld


BENCHMARK_VERSION = "roif_rda_1_first_endogenous_difference_v1"

FROZEN_ANALYZER_COMMIT = "0aee759"
FROZEN_ABSOLUTE_SCALE_FLOOR = 1e-9


@dataclass(frozen=True, slots=True)
class FirstEndogenousDifferenceConfig:
    """
    First exposure of the frozen endogenous difference analyzer.

    The analyzer receives only embodied Observation objects.

    Perturbation information is retained exclusively as experimenter-side
    ground truth for post-hoc evaluation.
    """

    duration: float = 5.0
    dt: float = 0.001
    baseline_duration: float = 1.0
    perturbation_time: float = 2.0
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
            raise ValueError("duration must be > 0.")

        if self.dt <= 0.0:
            raise ValueError("dt must be > 0.")

        if not 0.0 < self.baseline_duration < self.duration:
            raise ValueError(
                "baseline_duration must lie inside the run."
            )

        if not (
            self.baseline_duration
            < self.perturbation_time
            < self.duration
        ):
            raise ValueError(
                "perturbation_time must occur after baseline_duration "
                "and before the end of the run."
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


def _profile_to_record(
    profile: DifferenceProfile,
) -> dict[str, Any]:
    strongest = profile.strongest()

    return {
        "sequence_index": profile.sequence_index,
        "timestamp": profile.timestamp,
        "maximum_absolute_deviation": (
            profile.maximum_absolute_deviation
        ),
        "l2_deviation_norm": (
            profile.l2_deviation_norm
        ),
        "channel_deviations": [
            {
                "channel_name": deviation.channel_name,
                "current_value": deviation.current_value,
                "familiar_mean": deviation.familiar_mean,
                "familiar_scale": deviation.familiar_scale,
                "signed_deviation": deviation.signed_deviation,
                "absolute_deviation": deviation.absolute_deviation,
            }
            for deviation in profile.channel_deviations
        ],
        "strongest_channels": [
            {
                "rank": rank,
                "channel_name": deviation.channel_name,
                "signed_deviation": deviation.signed_deviation,
                "absolute_deviation": deviation.absolute_deviation,
            }
            for rank, deviation in enumerate(
                strongest,
                start=1,
            )
        ],
    }


def run_first_endogenous_difference(
    config: FirstEndogenousDifferenceConfig | None = None,
) -> dict[str, Any]:
    """
    First blind exposure of the frozen EndogenousDifferenceAnalyzer.

    Sequence:

        familiar embodied existence
            ->
        analyzer fit on prior observations only
            ->
        future observations analyzed blindly
            ->
        one later local perturbation in world ground truth
            ->
        DifferenceProfile trajectory
            ->
        post-hoc experimenter evaluation

    The analyzer itself never receives perturbation metadata.
    """

    if config is None:
        config = FirstEndogenousDifferenceConfig()

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

    analyzer = EndogenousDifferenceAnalyzer(
        absolute_scale_floor=(
            FROZEN_ABSOLUTE_SCALE_FLOOR
        )
    )

    # ---------------------------------------------------------
    # Phase 1: familiar embodied baseline.
    # ---------------------------------------------------------

    first_observation = body.observe()

    experience.append(
        first_observation
    )

    baseline_observations = [
        first_observation
    ]

    for step_index in range(
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

    analyzer.fit(
        baseline_observations
    )

    # ---------------------------------------------------------
    # Phase 2: blind endogenous difference analysis.
    # ---------------------------------------------------------

    profile_records: list[
        dict[str, Any]
    ] = []

    perturbation_profile: dict[
        str,
        Any,
    ] | None = None

    maximum_profile: dict[
        str,
        Any,
    ] | None = None

    maximum_profile_step: int | None = None

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

        profile = analyzer.analyze(
            observation
        )

        record = _profile_to_record(
            profile
        )

        record["experimenter_step_index"] = (
            step_index
        )

        profile_records.append(
            record
        )

        if step_index == perturbation_step:
            perturbation_profile = (
                dict(
                    record
                )
            )

        if (
            maximum_profile is None
            or record[
                "maximum_absolute_deviation"
            ]
            > maximum_profile[
                "maximum_absolute_deviation"
            ]
        ):
            maximum_profile = (
                dict(
                    record
                )
            )
            maximum_profile_step = (
                step_index
            )

    # ---------------------------------------------------------
    # Post-hoc evaluation only.
    # ---------------------------------------------------------

    if perturbation_profile is None:
        raise RuntimeError(
            "Perturbation profile was not recorded."
        )

    strongest_at_perturbation = (
        perturbation_profile[
            "strongest_channels"
        ]
    )

    top_10_at_perturbation = (
        strongest_at_perturbation[:10]
    )

    positive_count = sum(
        1
        for item in strongest_at_perturbation
        if item["signed_deviation"] > 0.0
    )

    negative_count = sum(
        1
        for item in strongest_at_perturbation
        if item["signed_deviation"] < 0.0
    )

    approximately_unchanged_count = sum(
        1
        for item in strongest_at_perturbation
        if np.isclose(
            item["signed_deviation"],
            0.0,
            atol=1e-12,
            rtol=0.0,
        )
    )

    perturbation_nonzero_count = sum(
        1
        for item in strongest_at_perturbation
        if item["absolute_deviation"] > 1e-12
    )

    result = {
        "benchmark": BENCHMARK_VERSION,
        "developmental_stage": "RDA-1",
        "experience": "first_endogenous_difference",
        "frozen_analyzer": {
            "commit": FROZEN_ANALYZER_COMMIT,
            "absolute_scale_floor": (
                FROZEN_ABSOLUTE_SCALE_FLOOR
            ),
            "parameters_modified_after_freeze": False,
        },
        "epistemic_boundary": {
            "analyzer_received_ground_truth": False,
            "analyzer_received_perturbation_time": False,
            "analyzer_received_perturbation_node": False,
            "analyzer_received_force_vector": False,
            "analyzer_received_world_coordinates": False,
            "analyzer_received_reward": False,
            "analyzer_received_goal": False,
            "analyzer_received_event_label": False,
            "analyzer_received_valence": False,
            "analyzer_received_body_part_labels": False,
        },
        "configuration": {
            "duration": (
                config.duration
            ),
            "dt": (
                config.dt
            ),
            "baseline_duration": (
                config.baseline_duration
            ),
            "baseline_sample_count": (
                len(
                    baseline_observations
                )
            ),
            "profile_count": (
                len(
                    profile_records
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
        },
        "perturbation_profile_summary": {
            "sequence_index": (
                perturbation_profile[
                    "sequence_index"
                ]
            ),
            "timestamp": (
                perturbation_profile[
                    "timestamp"
                ]
            ),
            "maximum_absolute_deviation": (
                perturbation_profile[
                    "maximum_absolute_deviation"
                ]
            ),
            "l2_deviation_norm": (
                perturbation_profile[
                    "l2_deviation_norm"
                ]
            ),
            "channel_count": (
                len(
                    perturbation_profile[
                        "channel_deviations"
                    ]
                )
            ),
            "nonzero_channel_count": (
                perturbation_nonzero_count
            ),
            "positive_signed_channel_count": (
                positive_count
            ),
            "negative_signed_channel_count": (
                negative_count
            ),
            "approximately_unchanged_channel_count": (
                approximately_unchanged_count
            ),
            "top_10_strongest_channels": (
                top_10_at_perturbation
            ),
        },
        "maximum_profile_summary": {
            "experimenter_step_index": (
                maximum_profile_step
            ),
            "sequence_index": (
                None
                if maximum_profile is None
                else maximum_profile[
                    "sequence_index"
                ]
            ),
            "timestamp": (
                None
                if maximum_profile is None
                else maximum_profile[
                    "timestamp"
                ]
            ),
            "maximum_absolute_deviation": (
                None
                if maximum_profile is None
                else maximum_profile[
                    "maximum_absolute_deviation"
                ]
            ),
            "l2_deviation_norm": (
                None
                if maximum_profile is None
                else maximum_profile[
                    "l2_deviation_norm"
                ]
            ),
        },
        "experience_summary": {
            "total_sample_count": (
                len(
                    experience
                )
            ),
            "experienced_duration": (
                experience.duration
            ),
        },
        "profile_records": (
            profile_records
        ),
    }

    result["profile_records_sha256"] = (
        _sha256(
            profile_records
        )
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
            "ROIF endogenous difference analyzer."
        )
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmark_results/"
            "rda_1_first_endogenous_difference_v1.json"
        ),
    )

    args = parser.parse_args()

    result = (
        run_first_endogenous_difference()
    )

    save_result(
        result=result,
        output_path=args.output,
    )

    summary = {
        "benchmark": (
            result["benchmark"]
        ),
        "frozen_analyzer_commit": (
            result["frozen_analyzer"][
                "commit"
            ]
        ),
        "perturbation_profile_summary": (
            result[
                "perturbation_profile_summary"
            ]
        ),
        "maximum_profile_summary": (
            result[
                "maximum_profile_summary"
            ]
        ),
        "profile_records_sha256": (
            result[
                "profile_records_sha256"
            ]
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
