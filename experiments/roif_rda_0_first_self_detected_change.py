from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from roif.development.experience_stream import ExperienceStream
from roif.development.familiar_state_change import (
    FamiliarStateChangeDetector,
)
from roif.worlds.tensegrity_3d_world import Tensegrity3DWorld


BENCHMARK_VERSION = "roif_rda_0_first_self_detected_change_v1"

FROZEN_DETECTOR_COMMIT = "28deb00"
FROZEN_THRESHOLD = 8.0
FROZEN_ABSOLUTE_SCALE_FLOOR = 1e-9


@dataclass(frozen=True, slots=True)
class FirstSelfDetectedChangeConfig:
    """
    Configuration for the first exposure of the frozen familiar-state
    detector to an embodied perturbation.

    Important:
        perturbation information belongs only to experimenter ground truth.
        It is never passed into FamiliarStateChangeDetector.
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


def run_first_self_detected_change(
    config: FirstSelfDetectedChangeConfig | None = None,
    *,
    return_internal: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], Any]:
    """
    First blind exposure of the frozen familiar-state detector.

    Experimental structure:

        1. ROIF body exists in self-stressed equilibrium.
        2. Initial proprioceptive experience is collected.
        3. Detector is fitted only on this earlier familiar experience.
        4. Detector is frozen for the remainder of the run.
        5. World later applies one local perturbation.
        6. Detector receives only Observation objects.
        7. Ground truth is consulted only after detection for evaluation.

    Detector never receives:
        - perturbation time,
        - perturbation node,
        - perturbation force,
        - world coordinates,
        - reward,
        - goal,
        - event label,
        - valence.
    """

    if config is None:
        config = FirstSelfDetectedChangeConfig()

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

    detector = FamiliarStateChangeDetector(
        threshold=FROZEN_THRESHOLD,
        absolute_scale_floor=(
            FROZEN_ABSOLUTE_SCALE_FLOOR
        ),
    )

    # ---------------------------------------------------------
    # Phase 1: familiar embodied existence.
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

    detector.fit(
        baseline_observations
    )

    # ---------------------------------------------------------
    # Phase 2: detector receives future experience blindly.
    # ---------------------------------------------------------

    detection_records: list[
        dict[str, Any]
    ] = []

    first_detection: dict[
        str,
        Any,
    ] | None = None
    first_detection_step: int | None = None
    first_detection_object = None

    maximum_score = 0.0
    maximum_score_timestamp: float | None = None

    false_positive_count = 0
    post_perturbation_detection_count = 0

    # baseline_steps observations already exist.
    # Continue from the following physics step.
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

        detection = detector.detect(
            observation
        )

        record = {
            "sequence_index": (
                detection.sequence_index
            ),
            "timestamp": (
                detection.timestamp
            ),
            "deviation_score": (
                detection.deviation_score
            ),
            "changed": (
                detection.changed
            ),
        }

        detection_records.append(
            record
        )

        if (
            detection.deviation_score
            > maximum_score
        ):
            maximum_score = (
                detection.deviation_score
            )

            maximum_score_timestamp = (
                detection.timestamp
            )

        if detection.changed:
            if first_detection is None:
                first_detection = dict(
                    record
                )
                first_detection_object = detection
                first_detection_step = step_index

            if (
                step_index
                < perturbation_step
            ):
                false_positive_count += 1
            else:
                post_perturbation_detection_count += 1

    # ---------------------------------------------------------
    # Evaluation.
    #
    # Ground truth is used here only after detector operation.
    # ---------------------------------------------------------

    detected = (
        first_detection is not None
    )

    first_detection_timestamp = (
        None
        if first_detection is None
        else float(
            first_detection["timestamp"]
        )
    )

    first_detection_before_perturbation = (
        detected
        and first_detection_step is not None
        and first_detection_step < perturbation_step
    )

    first_detection_at_or_after_perturbation = (
        detected
        and first_detection_step is not None
        and first_detection_step >= perturbation_step
    )

    detection_delay_steps = (
        None
        if not first_detection_at_or_after_perturbation
        else first_detection_step - perturbation_step
    )

    detection_delay = (
        None
        if detection_delay_steps is None
        else detection_delay_steps * config.dt
    )

    result = {
        "benchmark": BENCHMARK_VERSION,
        "developmental_stage": "RDA-0",
        "experience": "first_self_detected_change",
        "frozen_detector": {
            "commit": FROZEN_DETECTOR_COMMIT,
            "threshold": FROZEN_THRESHOLD,
            "absolute_scale_floor": (
                FROZEN_ABSOLUTE_SCALE_FLOOR
            ),
            "parameters_modified_after_freeze": False,
        },
        "epistemic_boundary": {
            "detector_received_ground_truth": False,
            "detector_received_perturbation_time": False,
            "detector_received_perturbation_node": False,
            "detector_received_force_vector": False,
            "detector_received_world_coordinates": False,
            "detector_received_reward": False,
            "detector_received_goal": False,
            "detector_received_event_label": False,
            "detector_received_valence": False,
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
        "detection_summary": {
            "detected_any_change": (
                detected
            ),
            "first_detection": (
                first_detection
            ),
            "first_detection_before_perturbation": (
                first_detection_before_perturbation
            ),
            "first_detection_at_or_after_perturbation": (
                first_detection_at_or_after_perturbation
            ),
            "detection_delay_steps": (
                detection_delay_steps
            ),
            "detection_delay": (
                detection_delay
            ),
            "false_positive_count_before_perturbation": (
                false_positive_count
            ),
            "post_perturbation_detection_count": (
                post_perturbation_detection_count
            ),
            "maximum_deviation_score": (
                float(
                    maximum_score
                )
            ),
            "maximum_score_timestamp": (
                maximum_score_timestamp
            ),
            "detector_threshold": (
                detector.threshold
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
        "detection_records": (
            detection_records
        ),
    }

    result["detection_records_sha256"] = (
        _sha256(
            detection_records
        )
    )

    result["run_sha256"] = _sha256(
        {
            key: value
            for key, value in result.items()
            if key != "run_sha256"
        }
    )

    if return_internal:
        if first_detection_object is None:
            raise RuntimeError(
                "No self-detected change was available for expression."
            )

        return result, first_detection_object

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
            "ROIF familiar-state change detector."
        )
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmark_results/"
            "rda_0_first_self_detected_change_v1.json"
        ),
    )

    args = parser.parse_args()

    result = (
        run_first_self_detected_change()
    )

    save_result(
        result=result,
        output_path=args.output,
    )

    summary = {
        "benchmark": (
            result["benchmark"]
        ),
        "frozen_detector_commit": (
            result["frozen_detector"][
                "commit"
            ]
        ),
        "detected_any_change": (
            result["detection_summary"][
                "detected_any_change"
            ]
        ),
        "first_detection": (
            result["detection_summary"][
                "first_detection"
            ]
        ),
        "first_detection_before_perturbation": (
            result["detection_summary"][
                "first_detection_before_perturbation"
            ]
        ),
        "first_detection_at_or_after_perturbation": (
            result["detection_summary"][
                "first_detection_at_or_after_perturbation"
            ]
        ),
        "detection_delay_steps": (
            result["detection_summary"][
                "detection_delay_steps"
            ]
        ),
        "detection_delay": (
            result["detection_summary"][
                "detection_delay"
            ]
        ),
        "false_positive_count_before_perturbation": (
            result["detection_summary"][
                "false_positive_count_before_perturbation"
            ]
        ),
        "post_perturbation_detection_count": (
            result["detection_summary"][
                "post_perturbation_detection_count"
            ]
        ),
        "maximum_deviation_score": (
            result["detection_summary"][
                "maximum_deviation_score"
            ]
        ),
        "maximum_score_timestamp": (
            result["detection_summary"][
                "maximum_score_timestamp"
            ]
        ),
        "detector_threshold": (
            result["detection_summary"][
                "detector_threshold"
            ]
        ),
        "detection_records_sha256": (
            result[
                "detection_records_sha256"
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
