from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from roif.development.dimensional_growth import (
    DimensionalGrowthAnalyzer,
)
from roif.development.endogenous_difference import (
    EndogenousDifferenceAnalyzer,
)
from roif.worlds.tensegrity_3d_world import Tensegrity3DWorld


BENCHMARK_VERSION = (
    "roif_rda_3_first_dimensional_growth_v1"
)

FROZEN_DIMENSIONAL_GROWTH_COMMIT = "31292ec"
FROZEN_DIFFERENCE_ANALYZER_COMMIT = "0aee759"

FROZEN_DIFFERENCE_SCALE_FLOOR = 1e-9

FROZEN_MINIMUM_RESIDUAL_VARIANCE_FRACTION = 0.01
FROZEN_MINIMUM_DIMENSION_FRACTION = 0.01
FROZEN_MINIMUM_SINGULAR_VALUE_RATIO = 1e-8
FROZEN_NUMERICAL_FLOOR = 1e-12


@dataclass(frozen=True, slots=True)
class FirstDimensionalGrowthConfig:
    """
    Первый слепой опыт размерностного роста.

    Исходное внутреннее представление строится только из собственного
    до-возмущающего опыта.

    Число его мерностей заранее не задаётся.

    После этого замороженный анализатор размерностного роста получает
    последующий опыт и оценивает, остаётся ли вне текущего представления
    устойчивая независимая структура.
    """

    duration: float = 5.0
    dt: float = 0.001

    baseline_duration: float = 1.0

    representation_start_time: float = 1.0
    representation_end_time: float = 2.0

    perturbation_time: float = 2.0

    assessment_start_time: float = 2.0
    assessment_end_time: float = 3.0

    perturbation_node: int = 0
    perturbation_force_x: float = 1.0
    perturbation_force_y: float = 0.0
    perturbation_force_z: float = 0.0

    representation_variance_fraction: float = 0.99

    def __post_init__(self) -> None:
        numeric = (
            self.duration,
            self.dt,
            self.baseline_duration,
            self.representation_start_time,
            self.representation_end_time,
            self.perturbation_time,
            self.assessment_start_time,
            self.assessment_end_time,
            self.perturbation_force_x,
            self.perturbation_force_y,
            self.perturbation_force_z,
            self.representation_variance_fraction,
        )

        if not all(
            np.isfinite(value)
            for value in numeric
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
            <= self.representation_start_time
            < self.representation_end_time
            <= self.perturbation_time
            <= self.assessment_start_time
            < self.assessment_end_time
            <= self.duration
        ):
            raise ValueError(
                "Required ordering: baseline <= representation "
                "< perturbation <= assessment <= duration."
            )

        if self.perturbation_node < 0:
            raise ValueError(
                "perturbation_node must be >= 0."
            )

        if not (
            0.0
            < self.representation_variance_fraction
            <= 1.0
        ):
            raise ValueError(
                "representation_variance_fraction "
                "must lie in (0, 1]."
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


def _profile_vector(
    profile,
    channel_names: tuple[str, ...],
) -> np.ndarray:
    values = {
        deviation.channel_name: (
            deviation.signed_deviation
        )
        for deviation in profile.channel_deviations
    }

    return np.asarray(
        [
            float(
                values[name]
            )
            for name in channel_names
        ],
        dtype=float,
    )


def _build_endogenous_representation(
    experience: np.ndarray,
    *,
    variance_fraction: float,
) -> tuple[
    np.ndarray,
    dict[str, Any],
]:
    """
    Построить исходное внутреннее пространство только из самого опыта.

    Никакое ожидаемое число мерностей не передаётся.

    Число направлений выбирается минимальным таким образом, чтобы они
    объясняли заданную долю собственной изменчивости до воздействия.
    """

    if experience.ndim != 2:
        raise ValueError(
            "experience must be two-dimensional."
        )

    if experience.shape[0] < 2:
        raise ValueError(
            "At least two samples are required."
        )

    centered = (
        experience
        - np.mean(
            experience,
            axis=0,
            keepdims=True,
        )
    )

    _u, singular_values, vt = np.linalg.svd(
        centered,
        full_matrices=False,
    )

    eigenvalues = (
        singular_values ** 2
    ) / max(
        experience.shape[0] - 1,
        1,
    )

    total = float(
        np.sum(
            eigenvalues
        )
    )

    if total <= 0.0:
        return (
            np.empty(
                (
                    experience.shape[1],
                    0,
                ),
                dtype=float,
            ),
            {
                "selected_dimension_count": 0,
                "total_variance": 0.0,
                "selected_variance_fraction": 0.0,
                "eigenvalues": (
                    eigenvalues.tolist()
                ),
            },
        )

    fractions = (
        eigenvalues
        / total
    )

    cumulative = np.cumsum(
        fractions
    )

    selected_dimension_count = int(
        np.searchsorted(
            cumulative,
            variance_fraction,
            side="left",
        )
        + 1
    )

    representation = (
        vt[
            :selected_dimension_count
        ].T
    )

    selected_variance_fraction = float(
        cumulative[
            selected_dimension_count - 1
        ]
    )

    return (
        np.asarray(
            representation,
            dtype=float,
        ),
        {
            "selected_dimension_count": (
                selected_dimension_count
            ),
            "total_variance": (
                total
            ),
            "selected_variance_fraction": (
                selected_variance_fraction
            ),
            "eigenvalues": (
                eigenvalues.tolist()
            ),
        },
    )


def run_first_dimensional_growth(
    config: FirstDimensionalGrowthConfig | None = None,
    *,
    return_internal: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], Any]:
    """
    Первый реальный слепой опыт размерностного роста.

    Критическая граница:

        число исходных внутренних мерностей не задаётся человеком;
        оно выводится из до-возмущающего собственного опыта.

    После этого замороженный анализатор оценивает только остаток,
    остающийся вне текущего представления.
    """

    if config is None:
        config = FirstDimensionalGrowthConfig()

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

    representation_start_step = _integer_step(
        time_value=config.representation_start_time,
        dt=config.dt,
        name="representation_start_time",
    )

    representation_end_step = _integer_step(
        time_value=config.representation_end_time,
        dt=config.dt,
        name="representation_end_time",
    )

    perturbation_step = _integer_step(
        time_value=config.perturbation_time,
        dt=config.dt,
        name="perturbation_time",
    )

    assessment_start_step = _integer_step(
        time_value=config.assessment_start_time,
        dt=config.dt,
        name="assessment_start_time",
    )

    assessment_end_step = _integer_step(
        time_value=config.assessment_end_time,
        dt=config.dt,
        name="assessment_end_time",
    )

    world = Tensegrity3DWorld()

    if config.perturbation_node >= world.node_count:
        raise ValueError(
            "perturbation_node is outside the body."
        )

    body = world.body(
        "tensegrity_body_0"
    )

    difference_analyzer = (
        EndogenousDifferenceAnalyzer(
            absolute_scale_floor=(
                FROZEN_DIFFERENCE_SCALE_FLOOR
            )
        )
    )

    growth_analyzer = (
        DimensionalGrowthAnalyzer(
            minimum_residual_variance_fraction=(
                FROZEN_MINIMUM_RESIDUAL_VARIANCE_FRACTION
            ),
            minimum_dimension_fraction=(
                FROZEN_MINIMUM_DIMENSION_FRACTION
            ),
            minimum_singular_value_ratio=(
                FROZEN_MINIMUM_SINGULAR_VALUE_RATIO
            ),
            numerical_floor=(
                FROZEN_NUMERICAL_FLOOR
            ),
        )
    )

    # ---------------------------------------------------------
    # Фаза 1. Знакомое состояние для анализатора различий.
    # ---------------------------------------------------------

    first_observation = body.observe()

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

        baseline_observations.append(
            body.observe()
        )

    difference_analyzer.fit(
        baseline_observations
    )

    channel_names = (
        difference_analyzer.channel_names
    )

    # ---------------------------------------------------------
    # Фаза 2. Эндогенное формирование исходного представления.
    # ---------------------------------------------------------

    representation_rows: list[
        np.ndarray
    ] = []

    assessment_rows: list[
        np.ndarray
    ] = []

    representation_step_indices: list[
        int
    ] = []

    assessment_step_indices: list[
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

        profile = (
            difference_analyzer.analyze(
                observation
            )
        )

        vector = _profile_vector(
            profile,
            channel_names,
        )

        if (
            representation_start_step
            <= step_index
            < perturbation_step
            and step_index
            <= representation_end_step
        ):
            representation_rows.append(
                vector.copy()
            )

            representation_step_indices.append(
                step_index
            )

        if (
            assessment_start_step
            <= step_index
            <= assessment_end_step
        ):
            assessment_rows.append(
                vector.copy()
            )

            assessment_step_indices.append(
                step_index
            )

    if len(representation_rows) < 2:
        raise RuntimeError(
            "Representation interval produced too few samples."
        )

    if len(assessment_rows) < 2:
        raise RuntimeError(
            "Assessment interval produced too few samples."
        )

    representation_experience = np.vstack(
        representation_rows
    )

    assessment_experience = np.vstack(
        assessment_rows
    )

    (
        representation,
        representation_summary,
    ) = _build_endogenous_representation(
        representation_experience,
        variance_fraction=(
            config.representation_variance_fraction
        ),
    )

    # ---------------------------------------------------------
    # Фаза 3. Первый слепой вопрос:
    #
    # достаточно ли пространства, которое возникло из прошлого опыта?
    # ---------------------------------------------------------

    assessment = growth_analyzer.assess(
        experience=assessment_experience,
        representation=representation,
    )

    residual_dimensions = [
        {
            "residual_index": (
                dimension.residual_index
            ),
            "eigenvalue": (
                dimension.eigenvalue
            ),
            "explained_residual_fraction": (
                dimension.explained_residual_fraction
            ),
            "cumulative_residual_fraction": (
                dimension.cumulative_residual_fraction
            ),
            "direction_sha256": _sha256(
                list(
                    dimension.direction
                )
            ),
        }
        for dimension
        in assessment.residual_dimensions
    ]

    result = {
        "benchmark": BENCHMARK_VERSION,
        "developmental_stage": "RDA-3",
        "experience": (
            "first_dimensional_growth_assessment"
        ),
        "frozen_components": {
            "dimensional_growth_analyzer_commit": (
                FROZEN_DIMENSIONAL_GROWTH_COMMIT
            ),
            "difference_analyzer_commit": (
                FROZEN_DIFFERENCE_ANALYZER_COMMIT
            ),
            "parameters_modified_after_freeze": False,
        },
        "epistemic_boundary": {
            "expected_dimension_count_supplied": False,
            "world_dimension_count_supplied": False,
            "topology_supplied": False,
            "node_mapping_supplied": False,
            "member_mapping_supplied": False,
            "semantic_dimension_labels_supplied": False,
            "reward_supplied": False,
            "goal_supplied": False,
            "event_label_supplied": False,
            "valence_supplied": False,
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
            "representation_start_time": (
                config.representation_start_time
            ),
            "representation_end_time": (
                config.representation_end_time
            ),
            "perturbation_time": (
                config.perturbation_time
            ),
            "assessment_start_time": (
                config.assessment_start_time
            ),
            "assessment_end_time": (
                config.assessment_end_time
            ),
            "representation_variance_fraction": (
                config.representation_variance_fraction
            ),
            "observed_channel_count": (
                len(
                    channel_names
                )
            ),
        },
        "endogenous_initial_representation": {
            "sample_count": int(
                representation_experience.shape[0]
            ),
            "selected_dimension_count": (
                representation_summary[
                    "selected_dimension_count"
                ]
            ),
            "selected_variance_fraction": (
                representation_summary[
                    "selected_variance_fraction"
                ]
            ),
            "total_variance": (
                representation_summary[
                    "total_variance"
                ]
            ),
        },
        "dimensional_growth_assessment": {
            "sample_count": (
                assessment.sample_count
            ),
            "observed_dimension": (
                assessment.observed_dimension
            ),
            "represented_dimension": (
                assessment.represented_dimension
            ),
            "residual_numerical_rank": (
                assessment.residual_numerical_rank
            ),
            "residual_effective_rank": (
                assessment.residual_effective_rank
            ),
            "residual_participation_ratio": (
                assessment.residual_participation_ratio
            ),
            "total_variance": (
                assessment.total_variance
            ),
            "represented_variance": (
                assessment.represented_variance
            ),
            "residual_variance": (
                assessment.residual_variance
            ),
            "residual_variance_fraction": (
                assessment.residual_variance_fraction
            ),
            "reconstruction_error_rms": (
                assessment.reconstruction_error_rms
            ),
            "normalized_reconstruction_error": (
                assessment.normalized_reconstruction_error
            ),
            "persistent_residual_dimension_count": (
                assessment.persistent_residual_dimension_count
            ),
            "growth_supported": (
                assessment.growth_supported
            ),
            "residual_dimensions": (
                residual_dimensions
            ),
        },
        "ground_truth_evaluation_only": {
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
        "representation_step_indices": (
            representation_step_indices
        ),
        "assessment_step_indices": (
            assessment_step_indices
        ),
    }

    result[
        "representation_experience_sha256"
    ] = _sha256(
        representation_experience.tolist()
    )

    result[
        "assessment_experience_sha256"
    ] = _sha256(
        assessment_experience.tolist()
    )

    result[
        "representation_basis_sha256"
    ] = _sha256(
        representation.tolist()
    )

    result["run_sha256"] = _sha256(
        {
            key: value
            for key, value in result.items()
            if key != "run_sha256"
        }
    )

    if return_internal:
        return result, assessment

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
            "Run the first blind ROIF dimensional growth assessment."
        )
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmark_results/"
            "rda_3_first_dimensional_growth_v1.json"
        ),
    )

    args = parser.parse_args()

    result = (
        run_first_dimensional_growth()
    )

    save_result(
        result=result,
        output_path=args.output,
    )

    assessment = result[
        "dimensional_growth_assessment"
    ]

    summary = {
        "benchmark": (
            result["benchmark"]
        ),
        "frozen_dimensional_growth_commit": (
            result["frozen_components"][
                "dimensional_growth_analyzer_commit"
            ]
        ),
        "observed_channel_count": (
            result["configuration"][
                "observed_channel_count"
            ]
        ),
        "initial_endogenous_dimension_count": (
            result[
                "endogenous_initial_representation"
            ][
                "selected_dimension_count"
            ]
        ),
        "represented_dimension": (
            assessment[
                "represented_dimension"
            ]
        ),
        "residual_numerical_rank": (
            assessment[
                "residual_numerical_rank"
            ]
        ),
        "residual_effective_rank": (
            assessment[
                "residual_effective_rank"
            ]
        ),
        "residual_participation_ratio": (
            assessment[
                "residual_participation_ratio"
            ]
        ),
        "residual_variance_fraction": (
            assessment[
                "residual_variance_fraction"
            ]
        ),
        "persistent_residual_dimension_count": (
            assessment[
                "persistent_residual_dimension_count"
            ]
        ),
        "growth_supported": (
            assessment[
                "growth_supported"
            ]
        ),
        "representation_experience_sha256": (
            result[
                "representation_experience_sha256"
            ]
        ),
        "assessment_experience_sha256": (
            result[
                "assessment_experience_sha256"
            ]
        ),
        "representation_basis_sha256": (
            result[
                "representation_basis_sha256"
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

