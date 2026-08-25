from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from experiments.roif_rda_0_first_self_detected_change import (
    FirstSelfDetectedChangeConfig,
)
from roif.development.endogenous_difference import (
    DifferenceProfile,
    EndogenousDifferenceAnalyzer,
)
from roif.development.experience_stream import ExperienceStream
from roif.development.familiar_state_change import (
    ChangeDetection,
    FamiliarStateChangeDetector,
)
from roif.worlds.tensegrity_3d_world import Tensegrity3DWorld


BENCHMARK_VERSION = (
    "roif_rda_1_first_detected_difference_v1"
)

FROZEN_DETECTOR_COMMIT = "28deb00"
FROZEN_DIFFERENCE_ANALYZER_COMMIT = "0aee759"

FROZEN_THRESHOLD = 8.0
FROZEN_ABSOLUTE_SCALE_FLOOR = 1e-9


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


def run_first_detected_difference(
    config: FirstSelfDetectedChangeConfig | None = None,
    *,
    return_internal: bool = False,
) -> (
    dict[str, Any]
    | tuple[
        dict[str, Any],
        ChangeDetection,
        DifferenceProfile,
    ]
):
    """
    Bind the first RDA-0 self-detected change to the
    RDA-1 DifferenceProfile produced from the same
    embodied Observation.

    Selection rule:

        first future Observation for which the frozen
        FamiliarStateChangeDetector returns changed=True.

    The selection rule never reads perturbation time,
    perturbation node, force vector, world coordinates,
    reward, goal, event labels, or valence.
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

    analyzer = EndogenousDifferenceAnalyzer(
        absolute_scale_floor=(
            FROZEN_ABSOLUTE_SCALE_FLOOR
        ),
    )

    # ---------------------------------------------------------
    # Phase 1:
    # one shared familiar embodied baseline.
    # ---------------------------------------------------------

    first_observation = body.observe()

    experience.append(
        first_observation
    )

    baseline_observations = [
        first_observation
    ]

    for _ in range(
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

    analyzer.fit(
        baseline_observations
    )

    # ---------------------------------------------------------
    # Phase 2:
    # both frozen mechanisms receive the same future
    # Observation objects.
    #
    # Selection is made ONLY by detector.changed.
    # ---------------------------------------------------------

    selected_detection: ChangeDetection | None = None
    selected_profile: DifferenceProfile | None = None

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

        profile = analyzer.analyze(
            observation
        )

        if detection.changed:
            selected_detection = detection
            selected_profile = profile
            break

    if (
        selected_detection is None
        or selected_profile is None
    ):
        raise RuntimeError(
            "No self-detected change was available "
            "for RDA-1 binding."
        )

    # ---------------------------------------------------------
    # Identity invariant:
    #
    # ChangeDetection and DifferenceProfile must describe
    # exactly the same experienced Observation.
    # ---------------------------------------------------------

    if (
        selected_detection.sequence_index
        != selected_profile.sequence_index
    ):
        raise RuntimeError(
            "Detection and DifferenceProfile "
            "sequence indices differ."
        )

    if not np.isclose(
        selected_detection.timestamp,
        selected_profile.timestamp,
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Detection and DifferenceProfile "
            "timestamps differ."
        )

    strongest = selected_profile.strongest(
        n=10
    )

    result: dict[str, Any] = {
        "benchmark": BENCHMARK_VERSION,
        "developmental_stage": "RDA-1",
        "experience": (
            "first_detected_endogenous_difference"
        ),
        "frozen_components": {
            "detector_commit": (
                FROZEN_DETECTOR_COMMIT
            ),
            "difference_analyzer_commit": (
                FROZEN_DIFFERENCE_ANALYZER_COMMIT
            ),
            "threshold": FROZEN_THRESHOLD,
            "absolute_scale_floor": (
                FROZEN_ABSOLUTE_SCALE_FLOOR
            ),
            "parameters_modified_after_freeze": False,
        },
        "selection_rule": {
            "rule": (
                "first_detection_changed_true"
            ),
            "uses_perturbation_time": False,
            "uses_perturbation_node": False,
            "uses_force_vector": False,
            "uses_world_coordinates": False,
            "uses_reward": False,
            "uses_goal": False,
            "uses_event_label": False,
            "uses_valence": False,
        },
        "matched_internal_state": {
            "sequence_index": (
                selected_profile.sequence_index
            ),
            "timestamp": (
                selected_profile.timestamp
            ),
            "detection_deviation_score": (
                selected_detection.deviation_score
            ),
            "detection_changed": (
                selected_detection.changed
            ),
            "maximum_absolute_deviation": (
                selected_profile.maximum_absolute_deviation
            ),
            "l2_deviation_norm": (
                selected_profile.l2_deviation_norm
            ),
            "channel_count": len(
                selected_profile.channel_deviations
            ),
            "top_10_strongest_channels": [
                {
                    "rank": rank,
                    "channel_name": (
                        deviation.channel_name
                    ),
                    "signed_deviation": (
                        deviation.signed_deviation
                    ),
                    "absolute_deviation": (
                        deviation.absolute_deviation
                    ),
                }
                for rank, deviation in enumerate(
                    strongest,
                    start=1,
                )
            ],
        },
        "identity_check": {
            "same_sequence_index": True,
            "same_timestamp": True,
        },
    }

    result["run_sha256"] = _sha256(
        {
            key: value
            for key, value in result.items()
            if key != "run_sha256"
        }
    )

    if return_internal:
        return (
            result,
            selected_detection,
            selected_profile,
        )

    return result


def main() -> None:
    result = run_first_detected_difference()

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
