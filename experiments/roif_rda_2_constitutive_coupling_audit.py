from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from roif.worlds.tensegrity_3d_world import Tensegrity3DWorld


BENCHMARK_VERSION = (
    "roif_rda_2_constitutive_coupling_audit_v1"
)

RDA_2_RESULT_COMMIT = "8e182d3"


@dataclass(frozen=True, slots=True)
class ConstitutiveCouplingAuditConfig:
    """
    External scientific audit of the RDA-2 grouping result.

    This audit is NOT part of the developmental input available to ROIF.

    It asks whether the observed pairing

        member_k_length
            <->
        member_k_axial_force

    can be explained by a direct local relation between member length
    and member axial force.
    """

    duration: float = 5.0
    dt: float = 0.001
    perturbation_time: float = 2.0
    perturbation_node: int = 0
    perturbation_force_x: float = 1.0
    perturbation_force_y: float = 0.0
    perturbation_force_z: float = 0.0

    fit_start_time: float = 2.0
    fit_end_time: float = 3.0

    def __post_init__(self) -> None:
        numeric_values = (
            self.duration,
            self.dt,
            self.perturbation_time,
            self.perturbation_force_x,
            self.perturbation_force_y,
            self.perturbation_force_z,
            self.fit_start_time,
            self.fit_end_time,
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
            < self.perturbation_time
            < self.duration
        ):
            raise ValueError(
                "perturbation_time must lie inside the run."
            )

        if not (
            0.0
            <= self.fit_start_time
            < self.fit_end_time
            <= self.duration
        ):
            raise ValueError(
                "fit interval must lie inside the run."
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


def _member_lengths(
    world: Tensegrity3DWorld,
) -> np.ndarray:
    """
    Obtain member lengths from the body's own observation channels.

    This avoids assuming private simulator internals.
    """

    observation = world.body(
        "tensegrity_body_0"
    ).observe()

    lengths: list[float] = []

    member_index = 0

    while True:
        channel_name = (
            f"member_{member_index}_length"
        )

        if channel_name not in observation.channels:
            break

        channel = observation.channels[
            channel_name
        ]

        lengths.append(
            float(
                channel.value
            )
        )

        member_index += 1

    if not lengths:
        raise RuntimeError(
            "No member length channels were found."
        )

    return np.asarray(
        lengths,
        dtype=float,
    )


def _member_forces(
    world: Tensegrity3DWorld,
) -> np.ndarray:
    forces = np.asarray(
        world.member_axial_forces(),
        dtype=float,
    )

    if forces.ndim != 1:
        raise RuntimeError(
            "member_axial_forces() must return a 1D array."
        )

    if not np.all(
        np.isfinite(
            forces
        )
    ):
        raise RuntimeError(
            "Non-finite member axial forces detected."
        )

    return forces


def _linear_fit_metrics(
    *,
    lengths: np.ndarray,
    forces: np.ndarray,
) -> dict[str, float | bool]:
    """
    Fit

        force ~= slope * length + intercept

    and report explanatory power and residual magnitude.

    No assumption is made here that the actual constitutive law must be
    linear; this is deliberately a simple falsification-oriented audit.
    """

    if lengths.ndim != 1:
        raise ValueError(
            "lengths must be one-dimensional."
        )

    if forces.ndim != 1:
        raise ValueError(
            "forces must be one-dimensional."
        )

    if lengths.shape != forces.shape:
        raise ValueError(
            "length and force series must have equal shape."
        )

    if lengths.size < 3:
        raise ValueError(
            "At least three samples are required."
        )

    if not np.all(
        np.isfinite(
            lengths
        )
    ):
        raise ValueError(
            "length series contains non-finite values."
        )

    if not np.all(
        np.isfinite(
            forces
        )
    ):
        raise ValueError(
            "force series contains non-finite values."
        )

    design = np.column_stack(
        [
            lengths,
            np.ones_like(
                lengths
            ),
        ]
    )

    coefficients, _, _, _ = np.linalg.lstsq(
        design,
        forces,
        rcond=None,
    )

    slope = float(
        coefficients[0]
    )

    intercept = float(
        coefficients[1]
    )

    predicted = (
        slope * lengths
        + intercept
    )

    residuals = (
        forces
        - predicted
    )

    ss_residual = float(
        np.sum(
            residuals ** 2
        )
    )

    force_mean = float(
        np.mean(
            forces
        )
    )

    ss_total = float(
        np.sum(
            (
                forces
                - force_mean
            ) ** 2
        )
    )

    if ss_total <= 1e-30:
        r_squared = (
            1.0
            if ss_residual <= 1e-30
            else 0.0
        )
    else:
        r_squared = float(
            1.0
            - ss_residual
            / ss_total
        )

    residual_rms = float(
        np.sqrt(
            np.mean(
                residuals ** 2
            )
        )
    )

    force_rms = float(
        np.sqrt(
            np.mean(
                forces ** 2
            )
        )
    )

    force_variation_rms = float(
        np.sqrt(
            np.mean(
                (
                    forces
                    - force_mean
                ) ** 2
            )
        )
    )

    relative_residual_to_force_rms = (
        residual_rms
        / force_rms
        if force_rms > 1e-30
        else 0.0
    )

    relative_residual_to_variation = (
        residual_rms
        / force_variation_rms
        if force_variation_rms > 1e-30
        else 0.0
    )

    length_variation = float(
        np.std(
            lengths,
            ddof=1,
        )
    )

    force_variation = float(
        np.std(
            forces,
            ddof=1,
        )
    )

    if (
        length_variation <= 1e-30
        or force_variation <= 1e-30
    ):
        pearson_correlation = 0.0
    else:
        pearson_correlation = float(
            np.corrcoef(
                lengths,
                forces,
            )[0, 1]
        )

    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "pearson_correlation": (
            pearson_correlation
        ),
        "residual_rms": residual_rms,
        "force_rms": force_rms,
        "force_variation_rms": (
            force_variation_rms
        ),
        "relative_residual_to_force_rms": (
            relative_residual_to_force_rms
        ),
        "relative_residual_to_force_variation": (
            relative_residual_to_variation
        ),
        "length_standard_deviation": (
            length_variation
        ),
        "force_standard_deviation": (
            force_variation
        ),
        "near_exact_linear_explanation": bool(
            r_squared >= 0.999999
            and relative_residual_to_variation
            <= 1e-3
        ),
    }


def run_constitutive_coupling_audit(
    config: ConstitutiveCouplingAuditConfig | None = None,
) -> dict[str, Any]:
    if config is None:
        config = ConstitutiveCouplingAuditConfig()

    total_steps = _integer_step(
        time_value=config.duration,
        dt=config.dt,
        name="duration",
    )

    perturbation_step = _integer_step(
        time_value=config.perturbation_time,
        dt=config.dt,
        name="perturbation_time",
    )

    fit_start_step = _integer_step(
        time_value=config.fit_start_time,
        dt=config.dt,
        name="fit_start_time",
    )

    fit_end_step = _integer_step(
        time_value=config.fit_end_time,
        dt=config.dt,
        name="fit_end_time",
    )

    world = Tensegrity3DWorld()

    if config.perturbation_node >= world.node_count:
        raise ValueError(
            "perturbation_node is outside the body."
        )

    initial_lengths = _member_lengths(
        world
    )

    initial_forces = _member_forces(
        world
    )

    if initial_lengths.shape != initial_forces.shape:
        raise RuntimeError(
            "Member length and force counts differ."
        )

    member_count = int(
        initial_lengths.size
    )

    fit_lengths: list[
        np.ndarray
    ] = []

    fit_forces: list[
        np.ndarray
    ] = []

    for step_index in range(
        1,
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

        if (
            fit_start_step
            <= step_index
            <= fit_end_step
        ):
            lengths = _member_lengths(
                world
            )

            forces = _member_forces(
                world
            )

            if lengths.shape != (
                member_count,
            ):
                raise RuntimeError(
                    "Member length channel count changed."
                )

            if forces.shape != (
                member_count,
            ):
                raise RuntimeError(
                    "Member force count changed."
                )

            fit_lengths.append(
                lengths.copy()
            )

            fit_forces.append(
                forces.copy()
            )

    if len(fit_lengths) < 3:
        raise RuntimeError(
            "Audit interval produced too few samples."
        )

    length_matrix = np.vstack(
        fit_lengths
    )

    force_matrix = np.vstack(
        fit_forces
    )

    member_results: list[
        dict[str, Any]
    ] = []

    for member_index in range(
        member_count
    ):
        metrics = _linear_fit_metrics(
            lengths=length_matrix[
                :,
                member_index,
            ],
            forces=force_matrix[
                :,
                member_index,
            ],
        )

        member_results.append(
            {
                "member_index": (
                    member_index
                ),
                "length_channel": (
                    f"member_{member_index}_length"
                ),
                "force_channel": (
                    f"member_{member_index}_axial_force"
                ),
                **metrics,
            }
        )

    near_exact_count = sum(
        1
        for item in member_results
        if item[
            "near_exact_linear_explanation"
        ]
    )

    r_squared_values = np.asarray(
        [
            float(
                item["r_squared"]
            )
            for item in member_results
        ],
        dtype=float,
    )

    relative_residual_values = np.asarray(
        [
            float(
                item[
                    "relative_residual_to_force_variation"
                ]
            )
            for item in member_results
        ],
        dtype=float,
    )

    correlation_values = np.asarray(
        [
            abs(
                float(
                    item[
                        "pearson_correlation"
                    ]
                )
            )
            for item in member_results
        ],
        dtype=float,
    )

    result = {
        "benchmark": BENCHMARK_VERSION,
        "audit_type": (
            "external_constitutive_coupling_audit"
        ),
        "audited_rda_2_commit": (
            RDA_2_RESULT_COMMIT
        ),
        "developmental_input_to_roif": False,
        "analyzer_modified": False,
        "configuration": {
            "duration": config.duration,
            "dt": config.dt,
            "perturbation_time": (
                config.perturbation_time
            ),
            "perturbation_node": (
                config.perturbation_node
            ),
            "perturbation_force": [
                config.perturbation_force_x,
                config.perturbation_force_y,
                config.perturbation_force_z,
            ],
            "fit_start_time": (
                config.fit_start_time
            ),
            "fit_end_time": (
                config.fit_end_time
            ),
            "fit_start_step": (
                fit_start_step
            ),
            "fit_end_step": (
                fit_end_step
            ),
            "fit_sample_count": (
                int(
                    length_matrix.shape[0]
                )
            ),
            "member_count": (
                member_count
            ),
        },
        "audit_summary": {
            "near_exact_linear_member_count": (
                near_exact_count
            ),
            "near_exact_linear_fraction": (
                near_exact_count
                / member_count
            ),
            "minimum_r_squared": float(
                np.min(
                    r_squared_values
                )
            ),
            "mean_r_squared": float(
                np.mean(
                    r_squared_values
                )
            ),
            "maximum_r_squared": float(
                np.max(
                    r_squared_values
                )
            ),
            "minimum_absolute_pearson_correlation": float(
                np.min(
                    correlation_values
                )
            ),
            "mean_absolute_pearson_correlation": float(
                np.mean(
                    correlation_values
                )
            ),
            "maximum_absolute_pearson_correlation": float(
                np.max(
                    correlation_values
                )
            ),
            "maximum_relative_residual_to_force_variation": float(
                np.max(
                    relative_residual_values
                )
            ),
            "mean_relative_residual_to_force_variation": float(
                np.mean(
                    relative_residual_values
                )
            ),
        },
        "member_results": (
            member_results
        ),
    }

    result[
        "member_results_sha256"
    ] = _sha256(
        member_results
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
            "Audit whether the first RDA-2 endogenous groups "
            "are explained by direct member length-force coupling."
        )
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmark_results/"
            "rda_2_constitutive_coupling_audit_v1.json"
        ),
    )

    args = parser.parse_args()

    result = (
        run_constitutive_coupling_audit()
    )

    save_result(
        result=result,
        output_path=args.output,
    )

    summary = {
        "benchmark": (
            result["benchmark"]
        ),
        "audited_rda_2_commit": (
            result[
                "audited_rda_2_commit"
            ]
        ),
        "audit_summary": (
            result[
                "audit_summary"
            ]
        ),
        "member_results_sha256": (
            result[
                "member_results_sha256"
            ]
        ),
        "run_sha256": (
            result[
                "run_sha256"
            ]
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
