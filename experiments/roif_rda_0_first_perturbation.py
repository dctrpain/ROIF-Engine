from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from roif.development.experience_stream import ExperienceStream
from roif.worlds.tensegrity_3d_world import Tensegrity3DWorld


BENCHMARK_VERSION = "roif_rda_0_first_perturbation_v1"


@dataclass(frozen=True, slots=True)
class FirstPerturbationConfig:
    duration: float = 5.0
    dt: float = 0.001
    observation_interval_steps: int = 1
    perturbation_time: float = 2.0
    perturbation_node: int = 0
    perturbation_force_x: float = 1.0
    perturbation_force_y: float = 0.0
    perturbation_force_z: float = 0.0

    def __post_init__(self) -> None:
        if self.duration <= 0.0:
            raise ValueError("duration must be > 0.")

        if self.dt <= 0.0:
            raise ValueError("dt must be > 0.")

        if self.observation_interval_steps <= 0:
            raise ValueError(
                "observation_interval_steps must be > 0."
            )

        if not 0.0 < self.perturbation_time < self.duration:
            raise ValueError(
                "perturbation_time must lie inside the run."
            )

        for value in (
            self.perturbation_force_x,
            self.perturbation_force_y,
            self.perturbation_force_z,
        ):
            if not np.isfinite(value):
                raise ValueError(
                    "perturbation force components must be finite."
                )


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(
        _canonical_json_bytes(value)
    ).hexdigest()


def _observation_record(sample) -> dict[str, Any]:
    observation = sample.observation

    return {
        "index": sample.index,
        "timestamp": observation.timestamp,
        "source_id": observation.provenance.source_id,
        "source_type": observation.provenance.source_type,
        "sequence_index": observation.provenance.sequence_index,
        "channels": {
            name: {
                "value": channel.value,
                "unit": channel.unit,
                "available": channel.available,
                "quality": channel.quality,
            }
            for name, channel in sorted(
                observation.channels.items()
            )
        },
        "metadata": dict(observation.metadata),
    }


def _contains_forbidden_semantics(
    records: list[dict[str, Any]],
) -> bool:
    forbidden = {
        "event",
        "event_type",
        "novelty",
        "reward",
        "goal",
        "valence",
        "meaning",
        "external_force",
        "external_force_x",
        "external_force_y",
        "external_force_z",
        "world_x",
        "world_y",
        "world_z",
        "node_0_x",
        "node_0_y",
        "node_0_z",
    }

    for record in records:
        if forbidden.intersection(record["channels"]):
            return True

    return False


def run_first_perturbation(
    config: FirstPerturbationConfig | None = None,
) -> dict[str, Any]:
    if config is None:
        config = FirstPerturbationConfig()

    world = Tensegrity3DWorld()
    body = world.body("tensegrity_body_0")

    experience = ExperienceStream(
        body_id=body.body_id
    )

    total_steps = int(round(config.duration / config.dt))
    perturbation_step = int(
        round(config.perturbation_time / config.dt)
    )

    if not np.isclose(
        total_steps * config.dt,
        config.duration,
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError(
            "duration must be an integer multiple of dt."
        )

    if not np.isclose(
        perturbation_step * config.dt,
        config.perturbation_time,
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError(
            "perturbation_time must be an integer multiple of dt."
        )

    initial_positions = world.positions.copy()
    initial_forces = world.member_axial_forces().copy()

    experience.append(body.observe())

    ground_truth_records: list[dict[str, Any]] = []

    for step_index in range(1, total_steps + 1):
        if step_index == perturbation_step:
            world.apply_external_force(
                node_index=config.perturbation_node,
                force=np.array(
                    [
                        config.perturbation_force_x,
                        config.perturbation_force_y,
                        config.perturbation_force_z,
                    ],
                    dtype=float,
                ),
            )

        world.step(config.dt)

        if step_index % config.observation_interval_steps == 0:
            experience.append(body.observe())

            ground_truth_records.append(
                {
                    "step_index": step_index,
                    "timestamp": world.time,
                    "positions": world.positions.tolist(),
                    "member_forces": (
                        world.member_axial_forces().tolist()
                    ),
                }
            )

    experience_records = [
        _observation_record(sample)
        for sample in experience
    ]

    final_positions = world.positions.copy()
    final_forces = world.member_axial_forces().copy()

    displacement_norms = np.linalg.norm(
        final_positions - initial_positions,
        axis=1,
    )

    force_changes = np.abs(
        final_forces - initial_forces
    )

    channel_names = sorted(
        experience.first.observation.channels
    )

    channel_series: dict[str, list[float]] = {
        name: []
        for name in channel_names
    }

    for sample in experience:
        for name in channel_names:
            value = sample.observation.channels[name].value
            channel_series[name].append(float(value))

    max_channel_excursions = {
        name: float(
            np.max(
                np.abs(
                    np.asarray(values)
                    - values[0]
                )
            )
        )
        for name, values in channel_series.items()
    }

    changed_channels = {
        name: excursion
        for name, excursion in max_channel_excursions.items()
        if excursion > 1e-12
    }

    result = {
        "benchmark": BENCHMARK_VERSION,
        "developmental_stage": "RDA-0",
        "experience": "first_perturbation",
        "interpretation_supplied_to_roif": False,
        "reward_supplied": False,
        "goal_supplied": False,
        "semantic_event_label_supplied": False,
        "configuration": {
            "duration": config.duration,
            "dt": config.dt,
            "observation_interval_steps": (
                config.observation_interval_steps
            ),
            "perturbation_time_ground_truth": (
                config.perturbation_time
            ),
            "perturbation_node_ground_truth": (
                config.perturbation_node
            ),
            "perturbation_force_ground_truth": [
                config.perturbation_force_x,
                config.perturbation_force_y,
                config.perturbation_force_z,
            ],
        },
        "experience_summary": {
            "sample_count": len(experience),
            "duration": experience.duration,
            "channel_count": len(channel_names),
            "changed_channel_count": len(changed_channels),
            "changed_channels": changed_channels,
            "ground_truth_or_semantic_leak_detected": (
                _contains_forbidden_semantics(
                    experience_records
                )
            ),
        },
        "physical_summary": {
            "initial_internal_force_norm": float(
                np.linalg.norm(initial_forces)
            ),
            "maximum_final_node_displacement": float(
                np.max(displacement_norms)
            ),
            "maximum_final_member_force_change": float(
                np.max(force_changes)
            ),
        },
        "experience_records": experience_records,
        "ground_truth_records": ground_truth_records,
    }

    result["experience_sha256"] = _sha256(
        experience_records
    )

    result["ground_truth_sha256"] = _sha256(
        ground_truth_records
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
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmark_results/"
            "rda_0_first_perturbation_v1.json"
        ),
    )

    args = parser.parse_args()

    result = run_first_perturbation()

    save_result(
        result=result,
        output_path=args.output,
    )

    summary = {
        "benchmark": result["benchmark"],
        "experience": result["experience"],
        "sample_count": (
            result["experience_summary"]["sample_count"]
        ),
        "changed_channel_count": (
            result["experience_summary"][
                "changed_channel_count"
            ]
        ),
        "ground_truth_or_semantic_leak_detected": (
            result["experience_summary"][
                "ground_truth_or_semantic_leak_detected"
            ]
        ),
        "maximum_final_node_displacement": (
            result["physical_summary"][
                "maximum_final_node_displacement"
            ]
        ),
        "maximum_final_member_force_change": (
            result["physical_summary"][
                "maximum_final_member_force_change"
            ]
        ),
        "experience_sha256": result[
            "experience_sha256"
        ],
        "run_sha256": result["run_sha256"],
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
