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


BENCHMARK_VERSION = "roif_rda_0_baseline_existence_v1"


@dataclass(frozen=True, slots=True)
class BaselineExistenceConfig:
    """
    Configuration for the first uninterrupted developmental experience.

    No perturbation, reward, goal, semantic event label, or externally
    supplied interpretation is introduced.
    """

    duration: float = 5.0
    dt: float = 0.001
    observation_interval_steps: int = 10

    def __post_init__(self) -> None:
        if self.duration <= 0.0:
            raise ValueError("duration must be > 0.")

        if self.dt <= 0.0:
            raise ValueError("dt must be > 0.")

        if self.observation_interval_steps <= 0:
            raise ValueError(
                "observation_interval_steps must be > 0."
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


def _observation_to_record(sample) -> dict[str, Any]:
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


def _ground_truth_to_record(
    *,
    step_index: int,
    world: Tensegrity3DWorld,
) -> dict[str, Any]:
    world_state = world.world_state()
    body_state = world.body(
        "tensegrity_body_0"
    ).ground_truth()

    return {
        "step_index": step_index,
        "timestamp": world.time,
        "world_state": {
            "values": dict(world_state.values),
            "metadata": dict(world_state.metadata),
        },
        "body_ground_truth": {
            "values": dict(body_state.values),
            "metadata": dict(body_state.metadata),
        },
    }


def _experience_contains_forbidden_ground_truth(
    experience_records: list[dict[str, Any]],
) -> bool:
    """
    Verify that directly experienced channels do not leak simulator-space
    coordinates or external semantic labels.
    """

    forbidden_exact = {
        "x",
        "y",
        "z",
        "world_x",
        "world_y",
        "world_z",
        "external_force",
        "external_force_x",
        "external_force_y",
        "external_force_z",
        "reward",
        "goal",
        "novelty",
        "event",
        "event_type",
        "valence",
        "meaning",
        "object_id",
    }

    for record in experience_records:
        channels = record["channels"]

        for name in channels:
            if name in forbidden_exact:
                return True

            if name.endswith("_world_x"):
                return True
            if name.endswith("_world_y"):
                return True
            if name.endswith("_world_z"):
                return True

    return False


def run_baseline_existence(
    config: BaselineExistenceConfig | None = None,
) -> dict[str, Any]:
    """
    Run ROIF RDA-0 Experience 0.

    The tensegrity body simply exists in deterministic self-stressed
    equilibrium.

    ROIF receives only the body's exposed proprioceptive observations.
    Ground truth is recorded separately for the experimenter.
    """

    if config is None:
        config = BaselineExistenceConfig()

    world = Tensegrity3DWorld()
    body = world.body("tensegrity_body_0")

    experience = ExperienceStream(
        body_id=body.body_id
    )

    ground_truth_records: list[dict[str, Any]] = []

    initial_positions = world.positions.copy()
    initial_member_forces = (
        world.member_axial_forces().copy()
    )

    total_steps_float = config.duration / config.dt
    total_steps = int(round(total_steps_float))

    if not np.isclose(
        total_steps * config.dt,
        config.duration,
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError(
            "duration must be an integer multiple of dt."
        )

    # First experienced state exists before the first world transition.
    experience.append(body.observe())

    ground_truth_records.append(
        _ground_truth_to_record(
            step_index=0,
            world=world,
        )
    )

    for step_index in range(1, total_steps + 1):
        world.step(config.dt)

        if (
            step_index
            % config.observation_interval_steps
            == 0
        ):
            experience.append(body.observe())

            ground_truth_records.append(
                _ground_truth_to_record(
                    step_index=step_index,
                    world=world,
                )
            )

    experience_records = [
        _observation_to_record(sample)
        for sample in experience
    ]

    final_positions = world.positions.copy()
    final_member_forces = world.member_axial_forces()

    maximum_node_displacement = float(
        np.max(
            np.linalg.norm(
                final_positions - initial_positions,
                axis=1,
            )
        )
    )

    maximum_member_force_change = float(
        np.max(
            np.abs(
                final_member_forces
                - initial_member_forces
            )
        )
    )

    experience_leaks_ground_truth = (
        _experience_contains_forbidden_ground_truth(
            experience_records
        )
    )

    result = {
        "benchmark": BENCHMARK_VERSION,
        "developmental_stage": "RDA-0",
        "experience": "baseline_existence",
        "interpretation_supplied_to_roif": False,
        "reward_supplied": False,
        "goal_supplied": False,
        "external_perturbation_supplied": False,
        "configuration": {
            "duration": config.duration,
            "dt": config.dt,
            "observation_interval_steps": (
                config.observation_interval_steps
            ),
            "physics_steps": total_steps,
        },
        "body": {
            "body_id": body.body_id,
            "node_count": world.node_count,
            "member_count": len(world.members),
            "strut_count": sum(
                member.kind == "strut"
                for member in world.members
            ),
            "cable_count": sum(
                member.kind == "cable"
                for member in world.members
            ),
        },
        "experience_summary": {
            "sample_count": len(experience),
            "duration": experience.duration,
            "channel_count": (
                len(experience.first.observation.channels)
                if experience.first is not None
                else 0
            ),
            "ground_truth_leak_detected": (
                experience_leaks_ground_truth
            ),
        },
        "physical_summary": {
            "initial_internal_force_norm": float(
                np.linalg.norm(initial_member_forces)
            ),
            "maximum_initial_nodal_force_norm": float(
                np.max(
                    np.linalg.norm(
                        Tensegrity3DWorld()
                        .nodal_internal_forces(),
                        axis=1,
                    )
                )
            ),
            "maximum_node_displacement": (
                maximum_node_displacement
            ),
            "maximum_member_force_change": (
                maximum_member_force_change
            ),
        },
        "experience_sha256": _sha256(
            experience_records
        ),
        "ground_truth_sha256": _sha256(
            ground_truth_records
        ),
        "experience_records": experience_records,
        "ground_truth_records": ground_truth_records,
    }

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
            "Run ROIF RDA-0 baseline existence: "
            "unperturbed continuous embodied experience."
        )
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--dt",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--observation-interval-steps",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmark_results/"
            "rda_0_baseline_existence_v1.json"
        ),
    )

    args = parser.parse_args()

    config = BaselineExistenceConfig(
        duration=args.duration,
        dt=args.dt,
        observation_interval_steps=(
            args.observation_interval_steps
        ),
    )

    result = run_baseline_existence(config)

    save_result(
        result=result,
        output_path=args.output,
    )

    summary = {
        "benchmark": result["benchmark"],
        "developmental_stage": (
            result["developmental_stage"]
        ),
        "experience": result["experience"],
        "sample_count": (
            result["experience_summary"]["sample_count"]
        ),
        "experienced_duration": (
            result["experience_summary"]["duration"]
        ),
        "ground_truth_leak_detected": (
            result["experience_summary"][
                "ground_truth_leak_detected"
            ]
        ),
        "initial_internal_force_norm": (
            result["physical_summary"][
                "initial_internal_force_norm"
            ]
        ),
        "maximum_node_displacement": (
            result["physical_summary"][
                "maximum_node_displacement"
            ]
        ),
        "maximum_member_force_change": (
            result["physical_summary"][
                "maximum_member_force_change"
            ]
        ),
        "experience_sha256": result[
            "experience_sha256"
        ],
        "ground_truth_sha256": result[
            "ground_truth_sha256"
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