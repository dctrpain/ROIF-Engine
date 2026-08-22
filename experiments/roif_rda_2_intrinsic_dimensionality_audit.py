from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from roif.development.endogenous_difference import (
    EndogenousDifferenceAnalyzer,
)
from roif.worlds.tensegrity_3d_world import Tensegrity3DWorld


BENCHMARK_VERSION = (
    "roif_rda_2_intrinsic_dimensionality_audit_v1"
)

REFERENCE_RDA_COMMIT = "96649c4"
FROZEN_DIFFERENCE_ANALYZER_COMMIT = "0aee759"
FROZEN_SCALE_FLOOR = 1e-9


@dataclass(frozen=True, slots=True)
class IntrinsicDimensionalityAuditConfig:
    duration: float = 5.0
    dt: float = 0.001

    baseline_duration: float = 1.0
    perturbation_time: float = 2.0

    analysis_start_time: float = 2.0
    analysis_end_time: float = 3.0

    perturbation_node: int = 0
    perturbation_force_x: float = 1.0
    perturbation_force_y: float = 0.0
    perturbation_force_z: float = 0.0

    temporal_variation_floor: float = 1e-12

    def __post_init__(self) -> None:
        numeric = (
            self.duration,
            self.dt,
            self.baseline_duration,
            self.perturbation_time,
            self.analysis_start_time,
            self.analysis_end_time,
            self.perturbation_force_x,
            self.perturbation_force_y,
            self.perturbation_force_z,
            self.temporal_variation_floor,
        )

        if not all(np.isfinite(value) for value in numeric):
            raise ValueError(
                "All numeric configuration values must be finite."
            )

        if self.duration <= 0.0:
            raise ValueError("duration must be > 0.")

        if self.dt <= 0.0:
            raise ValueError("dt must be > 0.")

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

        if not (
            0.0
            <= self.analysis_start_time
            < self.analysis_end_time
            <= self.duration
        ):
            raise ValueError(
                "Analysis interval must lie inside the run."
            )

        if self.analysis_start_time < self.perturbation_time:
            raise ValueError(
                "Analysis interval must not start before perturbation."
            )

        if self.perturbation_node < 0:
            raise ValueError(
                "perturbation_node must be >= 0."
            )

        if self.temporal_variation_floor < 0.0:
            raise ValueError(
                "temporal_variation_floor must be >= 0."
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


def _integer_step(
    *,
    time_value: float,
    dt: float,
    name: str,
) -> int:
    step = int(round(time_value / dt))

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


def _dimensions_for_variance(
    eigenvalues: np.ndarray,
    fraction: float,
) -> int:
    if eigenvalues.ndim != 1:
        raise ValueError(
            "eigenvalues must be one-dimensional."
        )

    total = float(np.sum(eigenvalues))

    if total <= 0.0:
        return 0

    cumulative = np.cumsum(eigenvalues) / total

    return int(
        np.searchsorted(
            cumulative,
            fraction,
            side="left",
        )
        + 1
    )


def _spectral_metrics(
    matrix: np.ndarray,
) -> dict[str, Any]:
    if matrix.ndim != 2:
        raise ValueError(
            "matrix must be two-dimensional."
        )

    if matrix.shape[0] < 2:
        raise ValueError(
            "At least two temporal samples are required."
        )

    if matrix.shape[1] < 1:
        raise ValueError(
            "At least one channel is required."
        )

    if not np.all(np.isfinite(matrix)):
        raise ValueError(
            "matrix contains non-finite values."
        )

    centered = (
        matrix
        - np.mean(
            matrix,
            axis=0,
            keepdims=True,
        )
    )

    singular_values = np.linalg.svd(
        centered,
        full_matrices=False,
        compute_uv=False,
    )

    eigenvalues = (
        singular_values ** 2
    ) / max(
        centered.shape[0] - 1,
        1,
    )

    eigenvalues = np.asarray(
        eigenvalues,
        dtype=float,
    )

    total = float(
        np.sum(eigenvalues)
    )

    if total <= 0.0:
        probabilities = np.zeros_like(
            eigenvalues
        )
    else:
        probabilities = (
            eigenvalues / total
        )

    positive_probabilities = probabilities[
        probabilities > 0.0
    ]

    if positive_probabilities.size:
        entropy = float(
            -np.sum(
                positive_probabilities
                * np.log(
                    positive_probabilities
                )
            )
        )

        effective_rank = float(
            np.exp(entropy)
        )
    else:
        entropy = 0.0
        effective_rank = 0.0

    denominator = float(
        np.sum(
            eigenvalues ** 2
        )
    )

    participation_ratio = (
        float(
            total ** 2
            / denominator
        )
        if denominator > 0.0
        else 0.0
    )

    maximum_eigenvalue = (
        float(eigenvalues[0])
        if eigenvalues.size
        else 0.0
    )

    stable_rank = (
        float(
            total
            / maximum_eigenvalue
        )
        if maximum_eigenvalue > 0.0
        else 0.0
    )

    tolerance = (
        max(centered.shape)
        * np.finfo(float).eps
        * (
            float(singular_values[0])
            if singular_values.size
            else 0.0
        )
    )

    numerical_rank = int(
        np.sum(
            singular_values > tolerance
        )
    )

    return {
        "sample_count": int(
            matrix.shape[0]
        ),
        "channel_count": int(
            matrix.shape[1]
        ),
        "numerical_rank": (
            numerical_rank
        ),
        "effective_rank": (
            effective_rank
        ),
        "participation_ratio": (
            participation_ratio
        ),
        "stable_rank": (
            stable_rank
        ),
        "spectral_entropy": (
            entropy
        ),
        "dimensions_for_90_percent_variance": (
            _dimensions_for_variance(
                eigenvalues,
                0.90,
            )
        ),
        "dimensions_for_95_percent_variance": (
            _dimensions_for_variance(
                eigenvalues,
                0.95,
            )
        ),
        "dimensions_for_99_percent_variance": (
            _dimensions_for_variance(
                eigenvalues,
                0.99,
            )
        ),
        "dimensions_for_99_9_percent_variance": (
            _dimensions_for_variance(
                eigenvalues,
                0.999,
            )
        ),
        "eigenvalues": (
            eigenvalues.tolist()
        ),
        "singular_values": (
            singular_values.tolist()
        ),
    }


def run_intrinsic_dimensionality_audit(
    config: IntrinsicDimensionalityAuditConfig | None = None,
) -> dict[str, Any]:
    if config is None:
        config = IntrinsicDimensionalityAuditConfig()

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

    analysis_start_step = _integer_step(
        time_value=config.analysis_start_time,
        dt=config.dt,
        name="analysis_start_time",
    )

    analysis_end_step = _integer_step(
        time_value=config.analysis_end_time,
        dt=config.dt,
        name="analysis_end_time",
    )

    world = Tensegrity3DWorld()

    if config.perturbation_node >= world.node_count:
        raise ValueError(
            "perturbation_node is outside the body."
        )

    body = world.body(
        "tensegrity_body_0"
    )

    analyzer = EndogenousDifferenceAnalyzer(
        absolute_scale_floor=(
            FROZEN_SCALE_FLOOR
        )
    )

    first_observation = body.observe()

    baseline_observations = [
        first_observation
    ]

    for _step_index in range(
        1,
        baseline_steps + 1,
    ):
        world.step(config.dt)

        baseline_observations.append(
            body.observe()
        )

    analyzer.fit(
        baseline_observations
    )

    channel_names = analyzer.channel_names

    analysis_rows: list[
        list[float]
    ] = []

    analysis_step_indices: list[
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

        world.step(config.dt)

        observation = body.observe()

        profile = analyzer.analyze(
            observation
        )

        if (
            analysis_start_step
            <= step_index
            <= analysis_end_step
        ):
            values = {
                deviation.channel_name: (
                    deviation.signed_deviation
                )
                for deviation
                in profile.channel_deviations
            }

            analysis_rows.append(
                [
                    float(
                        values[name]
                    )
                    for name in channel_names
                ]
            )

            analysis_step_indices.append(
                step_index
            )

    matrix = np.asarray(
        analysis_rows,
        dtype=float,
    )

    if matrix.shape[0] < 2:
        raise RuntimeError(
            "Analysis window produced too few samples."
        )

    temporal_standard_deviation = np.std(
        matrix,
        axis=0,
        ddof=1,
    )

    active_mask = (
        temporal_standard_deviation
        > config.temporal_variation_floor
    )

    active_indices = np.flatnonzero(
        active_mask
    )

    inactive_indices = np.flatnonzero(
        ~active_mask
    )

    active_channel_names = [
        channel_names[index]
        for index in active_indices
    ]

    inactive_channel_names = [
        channel_names[index]
        for index in inactive_indices
    ]

    if active_indices.size == 0:
        raise RuntimeError(
            "No temporally varying channels were found."
        )

    active_matrix = matrix[
        :,
        active_indices,
    ]

    active_temporal_std = (
        temporal_standard_deviation[
            active_indices
        ]
    )

    standardized_matrix = (
        active_matrix
        - np.mean(
            active_matrix,
            axis=0,
            keepdims=True,
        )
    ) / active_temporal_std

    raw_metrics = _spectral_metrics(
        active_matrix
    )

    standardized_metrics = _spectral_metrics(
        standardized_matrix
    )

    result = {
        "benchmark": BENCHMARK_VERSION,
        "audit_type": (
            "intrinsic_experienced_dimensionality"
        ),
        "reference_rda_commit": (
            REFERENCE_RDA_COMMIT
        ),
        "frozen_difference_analyzer_commit": (
            FROZEN_DIFFERENCE_ANALYZER_COMMIT
        ),
        "developmental_input_to_roif": False,
        "analyzer_modified": False,
        "topology_used": False,
        "semantic_group_labels_used": False,
        "configuration": {
            "duration": config.duration,
            "dt": config.dt,
            "baseline_duration": (
                config.baseline_duration
            ),
            "perturbation_time": (
                config.perturbation_time
            ),
            "analysis_start_time": (
                config.analysis_start_time
            ),
            "analysis_end_time": (
                config.analysis_end_time
            ),
            "analysis_start_step": (
                analysis_start_step
            ),
            "analysis_end_step": (
                analysis_end_step
            ),
            "analysis_sample_count": (
                int(matrix.shape[0])
            ),
            "observed_channel_count": (
                len(channel_names)
            ),
            "temporal_variation_floor": (
                config.temporal_variation_floor
            ),
        },
        "channel_activity": {
            "active_channel_count": (
                len(
                    active_channel_names
                )
            ),
            "inactive_channel_count": (
                len(
                    inactive_channel_names
                )
            ),
            "active_channel_names": (
                active_channel_names
            ),
            "inactive_channel_names": (
                inactive_channel_names
            ),
        },
        "raw_difference_space": (
            raw_metrics
        ),
        "temporally_standardized_difference_space": (
            standardized_metrics
        ),
        "analysis_step_indices": (
            analysis_step_indices
        ),
    }

    result["analysis_matrix_sha256"] = _sha256(
        matrix.tolist()
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
            "Measure intrinsic dimensionality of ROIF's "
            "experienced difference trajectory."
        )
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmark_results/"
            "rda_2_intrinsic_dimensionality_audit_v1.json"
        ),
    )

    args = parser.parse_args()

    result = run_intrinsic_dimensionality_audit()

    save_result(
        result=result,
        output_path=args.output,
    )

    summary = {
        "benchmark": result["benchmark"],
        "observed_channel_count": (
            result["configuration"][
                "observed_channel_count"
            ]
        ),
        "active_channel_count": (
            result["channel_activity"][
                "active_channel_count"
            ]
        ),
        "raw_difference_space": {
            key: value
            for key, value
            in result[
                "raw_difference_space"
            ].items()
            if key not in {
                "eigenvalues",
                "singular_values",
            }
        },
        "temporally_standardized_difference_space": {
            key: value
            for key, value
            in result[
                "temporally_standardized_difference_space"
            ].items()
            if key not in {
                "eigenvalues",
                "singular_values",
            }
        },
        "analysis_matrix_sha256": (
            result[
                "analysis_matrix_sha256"
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
