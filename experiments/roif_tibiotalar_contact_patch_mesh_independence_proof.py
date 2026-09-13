from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.surface_contact import (
    SurfaceContactCandidate,
    compliant_layer_compression,
    unilateral_normal_reaction,
)
from core.surface_mesh import SurfaceMesh


MM_TO_M = 1e-3

MESH_DIR = Path(
    r"C:\Users\DELL\OneDrive\Документы\roif-dev\body_parts_3d_api\meshes"
)

TIBIA_PATH = (
    MESH_DIR
    / "FJ3387_BP23960_FMA24477_Right tibia.obj"
)

TALUS_PATH = (
    MESH_DIR
    / "FJ3385_BP23520_FMA24482_Right talus.obj"
)

TALUS_SHIFT = np.array(
    [0.0, 0.0, 0.002],
    dtype=float,
)

# Synthetic mechanics parameters for this numerical proof only.
#
# IMPORTANT:
# These are NOT physiological cartilage parameters.
#
# The earlier point-contact proof used stiffness [N/m].
# Here the law is integrated over physical area, so the
# primitive coefficients are per-area:
#
#   stiffness_per_area [N / m^3]
#   damping_per_area   [N s / m^3]
#
# A cell with area A receives:
#
#   k_cell = stiffness_per_area * A
#   c_cell = damping_per_area * A
#
# Therefore refining the mesh does not automatically multiply
# the physical stiffness merely by creating more triangles.
REFERENCE_THICKNESS = 0.002
STIFFNESS_PER_AREA = 5.0e8
DAMPING_PER_AREA = 5.0e6

CANDIDATE_DISTANCE = REFERENCE_THICKNESS

# Synthetic closing velocity for the numerical invariance test.
CLOSING_SPEED = 0.01

# This is an experimental tolerance, not a physiological claim.
FORCE_RELATIVE_TOLERANCE = 0.10
MOMENT_RELATIVE_TOLERANCE = 0.10
COP_SHIFT_TOLERANCE_M = 0.001


@dataclass(frozen=True)
class IntegrationCell:
    vertices: np.ndarray
    parent_triangle: int


@dataclass(frozen=True)
class PatchResult:
    force: np.ndarray
    moment: np.ndarray
    center_of_pressure: np.ndarray
    total_loaded_area: float
    active_cell_count: int
    cell_count: int
    mean_normal_angle_deg: float
    max_normal_angle_deg: float
    weighted_normal_angle_deg: float
    fixed_normal_force: np.ndarray
    fixed_normal_moment: np.ndarray
    surface_normal_force: np.ndarray
    surface_normal_moment: np.ndarray
    talar_normal_force: np.ndarray
    talar_normal_moment: np.ndarray
    common_normal_force: np.ndarray
    common_normal_moment: np.ndarray


def triangle_area(
    triangle: np.ndarray,
) -> float:
    a, b, c = triangle

    return 0.5 * float(
        np.linalg.norm(
            np.cross(
                b - a,
                c - a,
            )
        )
    )


def triangle_centroid(
    triangle: np.ndarray,
) -> np.ndarray:
    return np.mean(
        triangle,
        axis=0,
    )


def subdivide_triangle(
    triangle: np.ndarray,
) -> list[np.ndarray]:
    a, b, c = triangle

    ab = 0.5 * (a + b)
    bc = 0.5 * (b + c)
    ca = 0.5 * (c + a)

    return [
        np.vstack([a, ab, ca]),
        np.vstack([ab, b, bc]),
        np.vstack([ca, bc, c]),
        np.vstack([ab, bc, ca]),
    ]


def relative_error(
    reference: np.ndarray,
    comparison: np.ndarray,
) -> float:
    denominator = float(
        np.linalg.norm(reference)
    )

    difference = float(
        np.linalg.norm(
            comparison - reference
        )
    )

    if denominator <= 1e-12:
        return difference

    return difference / denominator


def load_meshes() -> tuple[
    SurfaceMesh,
    SurfaceMesh,
]:
    tibia_raw = SurfaceMesh.from_obj(
        TIBIA_PATH
    )
    talus_raw = SurfaceMesh.from_obj(
        TALUS_PATH
    )

    tibia = SurfaceMesh(
        tibia_raw.vertices * MM_TO_M,
        tibia_raw.triangles,
    )

    talus = SurfaceMesh(
        (
            talus_raw.vertices * MM_TO_M
            + TALUS_SHIFT
        ),
        talus_raw.triangles,
    )

    return tibia, talus


def contact_parent_triangles(
    tibia: SurfaceMesh,
    talus: SurfaceMesh,
) -> tuple[
    list[int],
    np.ndarray,
]:
    candidates = (
        SurfaceContactCandidate
        .from_facing_nearby_meshes(
            tibia,
            talus,
            CANDIDATE_DISTANCE,
        )
    )

    active = [
        candidate
        for candidate in candidates
        if (
            candidate.contact_normal is not None
            and candidate.distance
            < REFERENCE_THICKNESS
        )
    ]

    if not active:
        raise RuntimeError(
            "no tibia-talus compliant-layer candidates found"
        )

    # One physical tibial surface triangle may appear in many
    # triangle-pair candidates.  It must enter the area integral
    # only once.
    parent_indices = sorted(
        {
            int(candidate.triangle_a)
            for candidate in active
        }
    )

    # Use the closest actual anatomical relation only to define
    # the direction of the imposed synthetic closing velocity.
    seed = min(
        active,
        key=lambda candidate: candidate.distance,
    )

    assert seed.contact_normal is not None

    reference_normal = np.asarray(
        seed.contact_normal,
        dtype=float,
    )

    reference_normal /= np.linalg.norm(
        reference_normal
    )

    return (
        parent_indices,
        reference_normal,
    )


def original_cells(
    mesh: SurfaceMesh,
    triangle_indices: list[int],
) -> list[IntegrationCell]:
    cells: list[IntegrationCell] = []

    for triangle_index in triangle_indices:
        triangle = np.vstack(
            mesh.triangle_vertices(
                triangle_index
            )
        )

        cells.append(
            IntegrationCell(
                vertices=triangle,
                parent_triangle=triangle_index,
            )
        )

    return cells


def refined_cells(
    cells: list[IntegrationCell],
) -> list[IntegrationCell]:
    refined: list[IntegrationCell] = []

    for cell in cells:
        for child in subdivide_triangle(
            cell.vertices
        ):
            refined.append(
                IntegrationCell(
                    vertices=child,
                    parent_triangle=cell.parent_triangle,
                )
            )

    return refined


def integrate_patch(
    cells: list[IntegrationCell],
    talus: SurfaceMesh,
    closing_velocity: np.ndarray,
    moment_origin: np.ndarray,
) -> PatchResult:
    total_force = np.zeros(
        3,
        dtype=float,
    )
    total_moment = np.zeros(
        3,
        dtype=float,
    )

    fixed_normal_force = np.zeros(
        3,
        dtype=float,
    )

    fixed_normal_moment = np.zeros(
        3,
        dtype=float,
    )

    surface_normal_force = np.zeros(
        3,
        dtype=float,
    )

    surface_normal_moment = np.zeros(
        3,
        dtype=float,
    )

    talar_normal_force = np.zeros(
        3,
        dtype=float,
    )

    talar_normal_moment = np.zeros(
        3,
        dtype=float,
    )

    common_normal_force = np.zeros(
        3,
        dtype=float,
    )

    common_normal_moment = np.zeros(
        3,
        dtype=float,
    )

    weighted_position = np.zeros(
        3,
        dtype=float,
    )

    total_force_weight = 0.0
    total_loaded_area = 0.0
    active_cell_count = 0

    normal_angles_deg: list[float] = []
    weighted_normal_angle_sum = 0.0

    closing_speed = float(
        np.linalg.norm(closing_velocity)
    )

    if closing_speed <= 1e-12:
        raise ValueError(
            "closing_velocity must be non-zero"
        )

    reference_normal = (
        -closing_velocity
        / closing_speed
    )

    for cell in cells:
        area = triangle_area(
            cell.vertices
        )

        if area <= 1e-16:
            continue

        tibia_point = triangle_centroid(
            cell.vertices
        )

        (
            talus_point,
            _talus_triangle,
            surface_distance,
        ) = talus.closest_point(
            tibia_point
        )

        compression = (
            compliant_layer_compression(
                surface_distance=surface_distance,
                reference_thickness=REFERENCE_THICKNESS,
            )
        )

        if compression <= 0.0:
            continue

        delta = (
            talus_point
            - tibia_point
        )

        distance = float(
            np.linalg.norm(delta)
        )

        if distance <= 1e-12:
            continue

        normal = delta / distance

        normal_alignment = float(
            np.clip(
                np.dot(
                    normal,
                    reference_normal,
                ),
                -1.0,
                1.0,
            )
        )

        normal_angle_deg = float(
            np.degrees(
                np.arccos(
                    normal_alignment
                )
            )
        )

        relative_normal_velocity = float(
            np.dot(
                closing_velocity,
                normal,
            )
        )

        # The existing ROIF unilateral law is retained.
        # Only its cell coefficients are scaled by physical area.
        cell_stiffness = (
            STIFFNESS_PER_AREA
            * area
        )

        cell_damping = (
            DAMPING_PER_AREA
            * area
        )

        reaction_magnitude = (
            unilateral_normal_reaction(
                compression=compression,
                relative_normal_velocity=relative_normal_velocity,
                stiffness=cell_stiffness,
                damping=cell_damping,
            )
        )

        if reaction_magnitude <= 0.0:
            continue

        # Force on talus.  Equal and opposite tibial force is
        # implicit; this proof compares the resultant patch load.
        force = (
            reaction_magnitude
            * normal
        )

        total_force += force

        total_moment += np.cross(
            tibia_point - moment_origin,
            force,
        )

        diagnostic_fixed_force = (
            reaction_magnitude
            * reference_normal
        )

        fixed_normal_force += (
            diagnostic_fixed_force
        )

        fixed_normal_moment += np.cross(
            tibia_point - moment_origin,
            diagnostic_fixed_force,
        )

        edge_1 = (
            cell.vertices[1]
            - cell.vertices[0]
        )
        edge_2 = (
            cell.vertices[2]
            - cell.vertices[0]
        )

        local_surface_normal = np.cross(
            edge_1,
            edge_2,
        )

        local_surface_normal_norm = float(
            np.linalg.norm(
                local_surface_normal
            )
        )

        if local_surface_normal_norm <= 1e-12:
            raise RuntimeError(
                "active integration cell has degenerate surface normal"
            )

        local_surface_normal = (
            local_surface_normal
            / local_surface_normal_norm
        )

        if (
            np.dot(
                local_surface_normal,
                reference_normal,
            )
            < 0.0
        ):
            local_surface_normal = (
                -local_surface_normal
            )

        diagnostic_surface_force = (
            reaction_magnitude
            * local_surface_normal
        )

        surface_normal_force += (
            diagnostic_surface_force
        )

        surface_normal_moment += np.cross(
            tibia_point - moment_origin,
            diagnostic_surface_force,
        )

        talar_surface_normal = (
            talus.triangle_normal(
                int(_talus_triangle)
            )
        )

        if talar_surface_normal is None:
            raise RuntimeError(
                "closest talar triangle has degenerate surface normal"
            )

        if (
            np.dot(
                talar_surface_normal,
                reference_normal,
            )
            < 0.0
        ):
            talar_surface_normal = (
                -talar_surface_normal
            )

        diagnostic_talar_force = (
            reaction_magnitude
            * talar_surface_normal
        )

        talar_normal_force += (
            diagnostic_talar_force
        )

        talar_normal_moment += np.cross(
            tibia_point - moment_origin,
            diagnostic_talar_force,
        )

        common_surface_normal = (
            local_surface_normal
            + talar_surface_normal
        )

        common_surface_normal_norm = float(
            np.linalg.norm(
                common_surface_normal
            )
        )

        if common_surface_normal_norm <= 1e-12:
            raise RuntimeError(
                "two-surface common normal is degenerate"
            )

        common_surface_normal = (
            common_surface_normal
            / common_surface_normal_norm
        )

        diagnostic_common_force = (
            reaction_magnitude
            * common_surface_normal
        )

        common_normal_force += (
            diagnostic_common_force
        )

        common_normal_moment += np.cross(
            tibia_point - moment_origin,
            diagnostic_common_force,
        )

        weighted_position += (
            reaction_magnitude
            * tibia_point
        )

        total_force_weight += (
            reaction_magnitude
        )

        normal_angles_deg.append(
            normal_angle_deg
        )

        weighted_normal_angle_sum += (
            reaction_magnitude
            * normal_angle_deg
        )

        total_loaded_area += area
        active_cell_count += 1

    if active_cell_count == 0:
        raise RuntimeError(
            "patch integration produced no active cells"
        )

    if total_force_weight <= 0.0:
        raise RuntimeError(
            "patch integration produced zero resultant load"
        )

    center_of_pressure = (
        weighted_position
        / total_force_weight
    )

    mean_normal_angle_deg = float(
        np.mean(
            normal_angles_deg
        )
    )

    max_normal_angle_deg = float(
        np.max(
            normal_angles_deg
        )
    )

    weighted_normal_angle_deg = float(
        weighted_normal_angle_sum
        / total_force_weight
    )

    return PatchResult(
        force=total_force,
        moment=total_moment,
        center_of_pressure=center_of_pressure,
        total_loaded_area=total_loaded_area,
        active_cell_count=active_cell_count,
        cell_count=len(cells),
        mean_normal_angle_deg=mean_normal_angle_deg,
        max_normal_angle_deg=max_normal_angle_deg,
        weighted_normal_angle_deg=weighted_normal_angle_deg,
        fixed_normal_force=fixed_normal_force,
        fixed_normal_moment=fixed_normal_moment,
        surface_normal_force=surface_normal_force,
        surface_normal_moment=surface_normal_moment,
        talar_normal_force=talar_normal_force,
        talar_normal_moment=talar_normal_moment,
        common_normal_force=common_normal_force,
        common_normal_moment=common_normal_moment,
    )


def print_result(
    name: str,
    result: PatchResult,
) -> None:
    print()
    print(name)
    print("-" * len(name))
    print(
        "cell_count:",
        result.cell_count,
    )
    print(
        "active_cell_count:",
        result.active_cell_count,
    )
    print(
        "loaded_area_m2:",
        result.total_loaded_area,
    )
    print(
        "force_N:",
        result.force.tolist(),
    )
    print(
        "force_magnitude_N:",
        float(
            np.linalg.norm(
                result.force
            )
        ),
    )
    print(
        "moment_Nm:",
        result.moment.tolist(),
    )
    print(
        "moment_magnitude_Nm:",
        float(
            np.linalg.norm(
                result.moment
            )
        ),
    )
    print(
        "center_of_pressure_m:",
        result.center_of_pressure.tolist(),
    )
    print(
        "mean_normal_angle_deg:",
        result.mean_normal_angle_deg,
    )
    print(
        "weighted_normal_angle_deg:",
        result.weighted_normal_angle_deg,
    )
    print(
        "max_normal_angle_deg:",
        result.max_normal_angle_deg,
    )
    print(
        "diagnostic_fixed_normal_force_N:",
        result.fixed_normal_force.tolist(),
    )
    print(
        "diagnostic_fixed_normal_force_magnitude_N:",
        float(
            np.linalg.norm(
                result.fixed_normal_force
            )
        ),
    )
    print(
        "diagnostic_fixed_normal_moment_Nm:",
        result.fixed_normal_moment.tolist(),
    )
    print(
        "diagnostic_fixed_normal_moment_magnitude_Nm:",
        float(
            np.linalg.norm(
                result.fixed_normal_moment
            )
        ),
    )
    print(
        "diagnostic_surface_normal_force_N:",
        result.surface_normal_force.tolist(),
    )
    print(
        "diagnostic_surface_normal_force_magnitude_N:",
        float(
            np.linalg.norm(
                result.surface_normal_force
            )
        ),
    )
    print(
        "diagnostic_surface_normal_moment_Nm:",
        result.surface_normal_moment.tolist(),
    )
    print(
        "diagnostic_surface_normal_moment_magnitude_Nm:",
        float(
            np.linalg.norm(
                result.surface_normal_moment
            )
        ),
    )
    print(
        "diagnostic_talar_normal_force_N:",
        result.talar_normal_force.tolist(),
    )
    print(
        "diagnostic_talar_normal_force_magnitude_N:",
        float(
            np.linalg.norm(
                result.talar_normal_force
            )
        ),
    )
    print(
        "diagnostic_talar_normal_moment_Nm:",
        result.talar_normal_moment.tolist(),
    )
    print(
        "diagnostic_talar_normal_moment_magnitude_Nm:",
        float(
            np.linalg.norm(
                result.talar_normal_moment
            )
        ),
    )
    print(
        "diagnostic_common_normal_force_N:",
        result.common_normal_force.tolist(),
    )
    print(
        "diagnostic_common_normal_force_magnitude_N:",
        float(
            np.linalg.norm(
                result.common_normal_force
            )
        ),
    )
    print(
        "diagnostic_common_normal_moment_Nm:",
        result.common_normal_moment.tolist(),
    )
    print(
        "diagnostic_common_normal_moment_magnitude_Nm:",
        float(
            np.linalg.norm(
                result.common_normal_moment
            )
        ),
    )


def main() -> None:
    print(
        "TIBIOTALAR DISTRIBUTED COMPLIANT "
        "CONTACT PATCH MESH-INDEPENDENCE PROOF"
    )
    print("=" * 70)

    print()
    print(
        "Boundary: numerical mechanics proof only."
    )
    print(
        "No physiological cartilage parameters are claimed."
    )
    print(
        "No new ROIF contact solver or production class is introduced."
    )

    tibia, talus = load_meshes()

    (
        parent_indices,
        reference_normal,
    ) = contact_parent_triangles(
        tibia,
        talus,
    )

    print()
    print(
        "anatomical_parent_triangle_count:",
        len(parent_indices),
    )
    print(
        "reference_contact_normal:",
        reference_normal.tolist(),
    )

    coarse_cells = original_cells(
        tibia,
        parent_indices,
    )

    refined_once_cells = refined_cells(
        coarse_cells
    )

    refined_twice_cells = refined_cells(
        refined_once_cells
    )

    refined_thrice_cells = refined_cells(
        refined_twice_cells
    )

    refined_fourth_cells = refined_cells(
        refined_thrice_cells
    )

    closing_velocity = (
        -CLOSING_SPEED
        * reference_normal
    )

    all_parent_vertices = np.vstack(
        [
            cell.vertices
            for cell in coarse_cells
        ]
    )

    moment_origin = np.mean(
        all_parent_vertices,
        axis=0,
    )

    coarse = integrate_patch(
        coarse_cells,
        talus,
        closing_velocity,
        moment_origin,
    )

    refined_once = integrate_patch(
        refined_once_cells,
        talus,
        closing_velocity,
        moment_origin,
    )

    refined_twice = integrate_patch(
        refined_twice_cells,
        talus,
        closing_velocity,
        moment_origin,
    )

    refined_thrice = integrate_patch(
        refined_thrice_cells,
        talus,
        closing_velocity,
        moment_origin,
    )

    refined_fourth = integrate_patch(
        refined_fourth_cells,
        talus,
        closing_velocity,
        moment_origin,
    )

    print_result(
        "COARSE ANATOMICAL TRIANGULATION",
        coarse,
    )

    print_result(
        "REFINED x4",
        refined_once,
    )

    print_result(
        "REFINED x16",
        refined_twice,
    )

    print_result(
        "REFINED x64",
        refined_thrice,
    )

    print_result(
        "REFINED x256",
        refined_fourth,
    )

    force_error_coarse_to_x4 = (
        relative_error(
            coarse.force,
            refined_once.force,
        )
    )

    force_error_x4_to_x16 = (
        relative_error(
            refined_once.force,
            refined_twice.force,
        )
    )

    force_error_x16_to_x64 = (
        relative_error(
            refined_twice.force,
            refined_thrice.force,
        )
    )

    moment_error_coarse_to_x4 = (
        relative_error(
            coarse.moment,
            refined_once.moment,
        )
    )

    moment_error_x4_to_x16 = (
        relative_error(
            refined_once.moment,
            refined_twice.moment,
        )
    )

    moment_error_x16_to_x64 = (
        relative_error(
            refined_twice.moment,
            refined_thrice.moment,
        )
    )

    fixed_force_error_coarse_to_x4 = (
        relative_error(
            coarse.fixed_normal_force,
            refined_once.fixed_normal_force,
        )
    )

    fixed_force_error_x4_to_x16 = (
        relative_error(
            refined_once.fixed_normal_force,
            refined_twice.fixed_normal_force,
        )
    )

    fixed_force_error_x16_to_x64 = (
        relative_error(
            refined_twice.fixed_normal_force,
            refined_thrice.fixed_normal_force,
        )
    )

    fixed_moment_error_coarse_to_x4 = (
        relative_error(
            coarse.fixed_normal_moment,
            refined_once.fixed_normal_moment,
        )
    )

    fixed_moment_error_x4_to_x16 = (
        relative_error(
            refined_once.fixed_normal_moment,
            refined_twice.fixed_normal_moment,
        )
    )

    fixed_moment_error_x16_to_x64 = (
        relative_error(
            refined_twice.fixed_normal_moment,
            refined_thrice.fixed_normal_moment,
        )
    )

    surface_force_error_coarse_to_x4 = (
        relative_error(
            coarse.surface_normal_force,
            refined_once.surface_normal_force,
        )
    )

    surface_force_error_x4_to_x16 = (
        relative_error(
            refined_once.surface_normal_force,
            refined_twice.surface_normal_force,
        )
    )

    surface_force_error_x16_to_x64 = (
        relative_error(
            refined_twice.surface_normal_force,
            refined_thrice.surface_normal_force,
        )
    )

    surface_moment_error_coarse_to_x4 = (
        relative_error(
            coarse.surface_normal_moment,
            refined_once.surface_normal_moment,
        )
    )

    surface_moment_error_x4_to_x16 = (
        relative_error(
            refined_once.surface_normal_moment,
            refined_twice.surface_normal_moment,
        )
    )

    surface_moment_error_x16_to_x64 = (
        relative_error(
            refined_twice.surface_normal_moment,
            refined_thrice.surface_normal_moment,
        )
    )

    talar_force_error_x16_to_x64 = (
        relative_error(
            refined_twice.talar_normal_force,
            refined_thrice.talar_normal_force,
        )
    )

    talar_moment_error_x16_to_x64 = (
        relative_error(
            refined_twice.talar_normal_moment,
            refined_thrice.talar_normal_moment,
        )
    )

    common_force_error_x16_to_x64 = (
        relative_error(
            refined_twice.common_normal_force,
            refined_thrice.common_normal_force,
        )
    )

    common_moment_error_x16_to_x64 = (
        relative_error(
            refined_twice.common_normal_moment,
            refined_thrice.common_normal_moment,
        )
    )

    common_force_error_x64_to_x256 = (
        relative_error(
            refined_thrice.common_normal_force,
            refined_fourth.common_normal_force,
        )
    )

    common_moment_error_x64_to_x256 = (
        relative_error(
            refined_thrice.common_normal_moment,
            refined_fourth.common_normal_moment,
        )
    )

    cop_shift_coarse_to_x4 = float(
        np.linalg.norm(
            refined_once.center_of_pressure
            - coarse.center_of_pressure
        )
    )

    cop_shift_x4_to_x16 = float(
        np.linalg.norm(
            refined_twice.center_of_pressure
            - refined_once.center_of_pressure
        )
    )

    cop_shift_x16_to_x64 = float(
        np.linalg.norm(
            refined_thrice.center_of_pressure
            - refined_twice.center_of_pressure
        )
    )

    cop_shift_x64_to_x256 = float(
        np.linalg.norm(
            refined_fourth.center_of_pressure
            - refined_thrice.center_of_pressure
        )
    )

    print()
    print("MESH REFINEMENT COMPARISON")
    print("-" * 70)

    print(
        "force_relative_error_coarse_to_x4:",
        force_error_coarse_to_x4,
    )
    print(
        "force_relative_error_x4_to_x16:",
        force_error_x4_to_x16,
    )
    print(
        "force_relative_error_x16_to_x64:",
        force_error_x16_to_x64,
    )
    print(
        "moment_relative_error_coarse_to_x4:",
        moment_error_coarse_to_x4,
    )
    print(
        "moment_relative_error_x4_to_x16:",
        moment_error_x4_to_x16,
    )
    print(
        "moment_relative_error_x16_to_x64:",
        moment_error_x16_to_x64,
    )
    print(
        "diagnostic_fixed_force_relative_error_coarse_to_x4:",
        fixed_force_error_coarse_to_x4,
    )
    print(
        "diagnostic_fixed_force_relative_error_x4_to_x16:",
        fixed_force_error_x4_to_x16,
    )
    print(
        "diagnostic_fixed_force_relative_error_x16_to_x64:",
        fixed_force_error_x16_to_x64,
    )
    print(
        "diagnostic_fixed_moment_relative_error_coarse_to_x4:",
        fixed_moment_error_coarse_to_x4,
    )
    print(
        "diagnostic_fixed_moment_relative_error_x4_to_x16:",
        fixed_moment_error_x4_to_x16,
    )
    print(
        "diagnostic_fixed_moment_relative_error_x16_to_x64:",
        fixed_moment_error_x16_to_x64,
    )
    print(
        "diagnostic_surface_force_relative_error_coarse_to_x4:",
        surface_force_error_coarse_to_x4,
    )
    print(
        "diagnostic_surface_force_relative_error_x4_to_x16:",
        surface_force_error_x4_to_x16,
    )
    print(
        "diagnostic_surface_force_relative_error_x16_to_x64:",
        surface_force_error_x16_to_x64,
    )
    print(
        "diagnostic_surface_moment_relative_error_coarse_to_x4:",
        surface_moment_error_coarse_to_x4,
    )
    print(
        "diagnostic_surface_moment_relative_error_x4_to_x16:",
        surface_moment_error_x4_to_x16,
    )
    print(
        "diagnostic_surface_moment_relative_error_x16_to_x64:",
        surface_moment_error_x16_to_x64,
    )
    print(
        "diagnostic_talar_force_relative_error_x16_to_x64:",
        talar_force_error_x16_to_x64,
    )
    print(
        "diagnostic_talar_moment_relative_error_x16_to_x64:",
        talar_moment_error_x16_to_x64,
    )
    print(
        "diagnostic_common_force_relative_error_x16_to_x64:",
        common_force_error_x16_to_x64,
    )
    print(
        "diagnostic_common_moment_relative_error_x16_to_x64:",
        common_moment_error_x16_to_x64,
    )
    print(
        "diagnostic_common_force_relative_error_x64_to_x256:",
        common_force_error_x64_to_x256,
    )
    print(
        "diagnostic_common_moment_relative_error_x64_to_x256:",
        common_moment_error_x64_to_x256,
    )
    print(
        "cop_shift_coarse_to_x4_m:",
        cop_shift_coarse_to_x4,
    )
    print(
        "cop_shift_x4_to_x16_m:",
        cop_shift_x4_to_x16,
    )
    print(
        "cop_shift_x16_to_x64_m:",
        cop_shift_x16_to_x64,
    )
    print(
        "cop_shift_x64_to_x256_m:",
        cop_shift_x64_to_x256,
    )


    # ------------------------------------------------------------------
    # ALTERNATE CONNECTIVITY DIAGNOSTIC
    #
    # This is deliberately separate from mesh_independence_supported.
    #
    # Six non-overlapping active-active tibial triangle pairs are
    # retriangulated by flipping their shared internal edge.
    #
    # The selected pairs satisfy:
    #   - both original cells are active in the present compliant state,
    #   - local face-to-face angle <= 2 degrees,
    #   - no original triangle belongs to more than one flip.
    #
    # This is NOT an exactly geometry-preserving remeshing operation:
    # the BodyParts3D surface is piecewise planar, so an edge flip across
    # non-coplanar adjacent faces introduces a small bounded geometric
    # perturbation.  The diagnostic therefore tests robustness of the
    # integrated articular resultant to controlled local connectivity
    # changes; it does not by itself establish general mesh independence.
    # ------------------------------------------------------------------

    alternate_flip_specs = [
        ((187, 189), (99, 838), (98, 111)),
        ((218, 236), (115, 121), (110, 118)),
        ((393, 573), (195, 280), (197, 179)),
        ((867, 871), (433, 435), (431, 436)),
        ((387, 388), (170, 196), (194, 171)),
        ((213, 382), (169, 170), (111, 194)),
    ]

    alternate_removed_triangles: set[int] = set()
    alternate_cells: list[IntegrationCell] = []

    for pair, shared_edge, opposite in alternate_flip_specs:
        triangle_i, triangle_j = pair
        edge_u, edge_v = shared_edge
        opposite_a, opposite_b = opposite

        alternate_removed_triangles.update(pair)

        point_u = tibia.vertices[edge_u]
        point_v = tibia.vertices[edge_v]
        point_a = tibia.vertices[opposite_a]
        point_b = tibia.vertices[opposite_b]

        alternate_cells.extend(
            [
                IntegrationCell(
                    vertices=np.vstack(
                        [point_a, point_b, point_u]
                    ),
                    parent_triangle=triangle_i,
                ),
                IntegrationCell(
                    vertices=np.vstack(
                        [point_a, point_v, point_b]
                    ),
                    parent_triangle=triangle_j,
                ),
            ]
        )

    alternate_patch_cells = [
        cell
        for cell in coarse_cells
        if cell.parent_triangle
        not in alternate_removed_triangles
    ]
    alternate_patch_cells.extend(alternate_cells)

    alternate_result = integrate_patch(
        alternate_patch_cells,
        talus,
        closing_velocity,
        moment_origin,
    )

    alternate_loaded_area_relative_difference = (
        abs(
            alternate_result.total_loaded_area
            - coarse.total_loaded_area
        )
        / coarse.total_loaded_area
    )

    alternate_force_relative_difference = float(
        np.linalg.norm(
            alternate_result.common_normal_force
            - coarse.common_normal_force
        )
        / max(
            float(np.linalg.norm(coarse.common_normal_force)),
            1.0e-15,
        )
    )

    alternate_moment_relative_difference = float(
        np.linalg.norm(
            alternate_result.common_normal_moment
            - coarse.common_normal_moment
        )
        / max(
            float(np.linalg.norm(coarse.common_normal_moment)),
            1.0e-15,
        )
    )

    alternate_cop_shift = float(
        np.linalg.norm(
            alternate_result.center_of_pressure
            - coarse.center_of_pressure
        )
    )

    print()
    print("ALTERNATE CONNECTIVITY DIAGNOSTIC")
    print("-" * 70)
    print(
        "alternate_flip_count:",
        len(alternate_flip_specs),
    )
    print(
        "alternate_replaced_triangle_count:",
        len(alternate_removed_triangles),
    )
    print(
        "alternate_active_cell_count:",
        alternate_result.active_cell_count,
    )
    print(
        "alternate_loaded_area_relative_difference:",
        alternate_loaded_area_relative_difference,
    )
    print(
        "alternate_common_force_relative_difference:",
        alternate_force_relative_difference,
    )
    print(
        "alternate_common_moment_relative_difference:",
        alternate_moment_relative_difference,
    )
    print(
        "alternate_cop_shift_m:",
        alternate_cop_shift,
    )
    print(
        "alternate_common_force_magnitude_N:",
        float(
            np.linalg.norm(
                alternate_result.common_normal_force
            )
        ),
    )
    print(
        "alternate_common_moment_magnitude_Nm:",
        float(
            np.linalg.norm(
                alternate_result.common_normal_moment
            )
        ),
    )


    # ------------------------------------------------------------------
    # TALAR-SIDE CONNECTIVITY DIAGNOSTIC
    #
    # This diagnostic perturbs the second articular surface itself.
    #
    # Three non-overlapping active-active talar edge flips were selected
    # before evaluating their mechanical result.  Selection was limited
    # to low-dihedral candidates with bounded cross-plane geometric
    # deviation (<= 0.25 mm in this state).
    #
    # As with the tibial diagnostic above, these flips are NOT exactly
    # geometry-preserving remeshing: the original surface is piecewise
    # planar.  This therefore tests robustness to controlled local
    # connectivity perturbation, not general mesh independence.
    # ------------------------------------------------------------------

    talar_flip_specs = [
        ((692, 1255), (340, 554), (253, 690)),
        ((1376, 1378), (689, 691), (631, 687)),
        ((1375, 1264), (631, 690), (691, 630)),
    ]

    alternate_talus_triangles = np.array(
        talus.triangles,
        copy=True,
    )

    def talus_unit_normal(
        triangle_vertices: np.ndarray,
    ) -> np.ndarray:
        point_a, point_b, point_c = talus.vertices[
            np.asarray(
                triangle_vertices,
                dtype=int,
            )
        ]

        normal = np.cross(
            point_b - point_a,
            point_c - point_a,
        )

        normal_norm = float(
            np.linalg.norm(normal)
        )

        if normal_norm <= 1.0e-15:
            raise ValueError(
                "Degenerate talar triangle during "
                "alternate-connectivity diagnostic."
            )

        return normal / normal_norm

    for (
        pair,
        shared_edge,
        opposite,
    ) in talar_flip_specs:
        triangle_i, triangle_j = pair
        edge_u, edge_v = shared_edge
        opposite_a, opposite_b = opposite

        old_normal_i = talus_unit_normal(
            alternate_talus_triangles[
                triangle_i
            ]
        )
        old_normal_j = talus_unit_normal(
            alternate_talus_triangles[
                triangle_j
            ]
        )

        local_reference_normal = (
            old_normal_i + old_normal_j
        )
        local_reference_norm = float(
            np.linalg.norm(
                local_reference_normal
            )
        )

        if local_reference_norm <= 1.0e-15:
            raise ValueError(
                "Opposed local talar normals during "
                "alternate-connectivity diagnostic."
            )

        local_reference_normal /= (
            local_reference_norm
        )

        new_triangle_i = np.array(
            [
                opposite_a,
                opposite_b,
                edge_u,
            ],
            dtype=int,
        )

        new_triangle_j = np.array(
            [
                opposite_a,
                edge_v,
                opposite_b,
            ],
            dtype=int,
        )

        if (
            float(
                np.dot(
                    talus_unit_normal(
                        new_triangle_i
                    ),
                    local_reference_normal,
                )
            )
            < 0.0
        ):
            new_triangle_i[[1, 2]] = (
                new_triangle_i[[2, 1]]
            )

        if (
            float(
                np.dot(
                    talus_unit_normal(
                        new_triangle_j
                    ),
                    local_reference_normal,
                )
            )
            < 0.0
        ):
            new_triangle_j[[1, 2]] = (
                new_triangle_j[[2, 1]]
            )

        alternate_talus_triangles[
            triangle_i
        ] = new_triangle_i

        alternate_talus_triangles[
            triangle_j
        ] = new_triangle_j

    alternate_talus = SurfaceMesh(
        np.array(
            talus.vertices,
            copy=True,
        ),
        alternate_talus_triangles,
    )

    talar_alternate_result = integrate_patch(
        coarse_cells,
        alternate_talus,
        closing_velocity,
        moment_origin,
    )

    talar_loaded_area_relative_difference = (
        abs(
            talar_alternate_result.total_loaded_area
            - coarse.total_loaded_area
        )
        / coarse.total_loaded_area
    )

    talar_force_relative_difference = float(
        np.linalg.norm(
            talar_alternate_result.common_normal_force
            - coarse.common_normal_force
        )
        / max(
            float(
                np.linalg.norm(
                    coarse.common_normal_force
                )
            ),
            1.0e-15,
        )
    )

    talar_moment_relative_difference = float(
        np.linalg.norm(
            talar_alternate_result.common_normal_moment
            - coarse.common_normal_moment
        )
        / max(
            float(
                np.linalg.norm(
                    coarse.common_normal_moment
                )
            ),
            1.0e-15,
        )
    )

    talar_cop_shift = float(
        np.linalg.norm(
            talar_alternate_result.center_of_pressure
            - coarse.center_of_pressure
        )
    )

    print()
    print("TALAR-SIDE CONNECTIVITY DIAGNOSTIC")
    print("-" * 70)
    print(
        "talar_flip_count:",
        len(talar_flip_specs),
    )
    print(
        "talar_replaced_triangle_count:",
        2 * len(talar_flip_specs),
    )
    print(
        "talar_active_cell_count:",
        talar_alternate_result.active_cell_count,
    )
    print(
        "talar_loaded_area_relative_difference:",
        talar_loaded_area_relative_difference,
    )
    print(
        "talar_common_force_relative_difference:",
        talar_force_relative_difference,
    )
    print(
        "talar_common_moment_relative_difference:",
        talar_moment_relative_difference,
    )
    print(
        "talar_cop_shift_m:",
        talar_cop_shift,
    )
    print(
        "talar_common_force_magnitude_N:",
        float(
            np.linalg.norm(
                talar_alternate_result.common_normal_force
            )
        ),
    )
    print(
        "talar_common_moment_magnitude_Nm:",
        float(
            np.linalg.norm(
                talar_alternate_result.common_normal_moment
            )
        ),
    )

    # ------------------------------------------------------------------
    # BILATERAL CONNECTIVITY DIAGNOSTIC
    #
    # Apply the already-defined tibial alternate integration cells and
    # the controlled talar connectivity perturbation simultaneously.
    #
    # This is the strongest connectivity-robustness diagnostic in this
    # experiment, but it remains state-specific and must not be promoted
    # to a general claim of mesh independence.
    # ------------------------------------------------------------------

    bilateral_result = integrate_patch(
        alternate_patch_cells,
        alternate_talus,
        closing_velocity,
        moment_origin,
    )

    bilateral_loaded_area_relative_difference = (
        abs(
            bilateral_result.total_loaded_area
            - coarse.total_loaded_area
        )
        / coarse.total_loaded_area
    )

    bilateral_force_relative_difference = float(
        np.linalg.norm(
            bilateral_result.common_normal_force
            - coarse.common_normal_force
        )
        / max(
            float(
                np.linalg.norm(
                    coarse.common_normal_force
                )
            ),
            1.0e-15,
        )
    )

    bilateral_moment_relative_difference = float(
        np.linalg.norm(
            bilateral_result.common_normal_moment
            - coarse.common_normal_moment
        )
        / max(
            float(
                np.linalg.norm(
                    coarse.common_normal_moment
                )
            ),
            1.0e-15,
        )
    )

    bilateral_cop_shift = float(
        np.linalg.norm(
            bilateral_result.center_of_pressure
            - coarse.center_of_pressure
        )
    )

    print()
    print("BILATERAL CONNECTIVITY DIAGNOSTIC")
    print("-" * 70)
    print(
        "bilateral_tibial_flip_count:",
        len(alternate_flip_specs),
    )
    print(
        "bilateral_talar_flip_count:",
        len(talar_flip_specs),
    )
    print(
        "bilateral_active_cell_count:",
        bilateral_result.active_cell_count,
    )
    print(
        "bilateral_loaded_area_relative_difference:",
        bilateral_loaded_area_relative_difference,
    )
    print(
        "bilateral_common_force_relative_difference:",
        bilateral_force_relative_difference,
    )
    print(
        "bilateral_common_moment_relative_difference:",
        bilateral_moment_relative_difference,
    )
    print(
        "bilateral_cop_shift_m:",
        bilateral_cop_shift,
    )
    print(
        "bilateral_common_force_magnitude_N:",
        float(
            np.linalg.norm(
                bilateral_result.common_normal_force
            )
        ),
    )
    print(
        "bilateral_common_moment_magnitude_Nm:",
        float(
            np.linalg.norm(
                bilateral_result.common_normal_moment
            )
        ),
    )

    # The strongest evidence is convergence:
    # refinement should reduce discretization sensitivity.
    converging_force = (
        force_error_x4_to_x16
        <= force_error_coarse_to_x4
        and force_error_x16_to_x64
        <= force_error_x4_to_x16
    )

    converging_moment = (
        moment_error_x4_to_x16
        <= moment_error_coarse_to_x4
        and moment_error_x16_to_x64
        <= moment_error_x4_to_x16
    )

    final_within_tolerance = (
        force_error_x16_to_x64
        <= FORCE_RELATIVE_TOLERANCE
        and moment_error_x16_to_x64
        <= MOMENT_RELATIVE_TOLERANCE
        and cop_shift_x16_to_x64
        <= COP_SHIFT_TOLERANCE_M
    )

    mesh_independence_supported = (
        converging_force
        and converging_moment
        and final_within_tolerance
    )

    # Separate diagnostic verdict for the two-surface common-normal
    # formulation only.  This does NOT change the original negative
    # mesh-independence verdict above and is not a general claim of
    # arbitrary-remeshing invariance.
    common_normal_refinement_supported = (
        common_force_error_x64_to_x256
        <= common_force_error_x16_to_x64
        and common_moment_error_x64_to_x256
        <= common_moment_error_x16_to_x64
        and common_force_error_x64_to_x256
        <= FORCE_RELATIVE_TOLERANCE
        and common_moment_error_x64_to_x256
        <= MOMENT_RELATIVE_TOLERANCE
        and cop_shift_x64_to_x256
        <= COP_SHIFT_TOLERANCE_M
    )

    print()
    print(
        "converging_force:",
        converging_force,
    )
    print(
        "converging_moment:",
        converging_moment,
    )
    print(
        "final_within_tolerance:",
        final_within_tolerance,
    )
    print(
        "mesh_independence_supported:",
        mesh_independence_supported,
    )
    print(
        "common_normal_refinement_supported:",
        common_normal_refinement_supported,
    )

    if not mesh_independence_supported:
        raise SystemExit(1)


if __name__ == "__main__":
    main()






