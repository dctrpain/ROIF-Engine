from __future__ import annotations

import json
from itertools import combinations

import numpy as np

from core.distance_constraint import DistanceConstraint
from core.network import Network
from core.node import Node
from core.solver import Solver


def centroid(nodes: list[Node]) -> np.ndarray:
    masses = np.asarray(
        [node.mass for node in nodes],
        dtype=float,
    )
    positions = np.vstack(
        [node.position for node in nodes]
    )

    return (
        np.sum(
            positions * masses[:, None],
            axis=0,
        )
        / np.sum(masses)
    )


def pair_distances(
    nodes: list[Node],
) -> dict[tuple[int, int], float]:
    result: dict[tuple[int, int], float] = {}

    for i, j in combinations(
        range(len(nodes)),
        2,
    ):
        result[(i, j)] = float(
            np.linalg.norm(
                nodes[j].position
                - nodes[i].position
            )
        )

    return result


def make_structure(
    *,
    prefix: str,
    origin: np.ndarray,
    fixed: bool,
) -> list[Node]:
    local_positions = (
        np.array(
            [0.0, 0.0, 0.0],
            dtype=float,
        ),
        np.array(
            [0.10, 0.0, 0.0],
            dtype=float,
        ),
        np.array(
            [0.0, 0.04, 0.0],
            dtype=float,
        ),
        np.array(
            [0.0, 0.0, 0.03],
            dtype=float,
        ),
    )

    return [
        Node(
            position=origin
            + local_position,
            mass=0.25,
            fixed=fixed,
            node_id=f"{prefix}_{index}",
        )
        for index, local_position
        in enumerate(local_positions)
    ]


def add_shape_constraints(
    network: Network,
    nodes: list[Node],
    prefix: str,
) -> None:
    for i, j in combinations(
        range(len(nodes)),
        2,
    ):
        network.add_constraint(
            DistanceConstraint(
                name=(
                    f"{prefix}_DISTANCE_"
                    f"{i}_{j}"
                ),
                node_a=nodes[i],
                node_b=nodes[j],
                target_length=float(
                    np.linalg.norm(
                        nodes[j].position
                        - nodes[i].position
                    )
                ),
                mode="equality",
                compliance=0.0,
                damping=0.0,
            )
        )


def attachment_position(
    nodes: list[Node],
    weights: np.ndarray,
) -> np.ndarray:
    result = np.zeros(
        3,
        dtype=float,
    )

    for node, weight in zip(
        nodes,
        weights,
    ):
        result += (
            float(weight)
            * node.position
        )

    return result


def attachment_velocity(
    nodes: list[Node],
    weights: np.ndarray,
) -> np.ndarray:
    result = np.zeros(
        3,
        dtype=float,
    )

    for node, weight in zip(
        nodes,
        weights,
    ):
        result += (
            float(weight)
            * node.velocity
        )

    return result


def distribute_attachment_force(
    nodes: list[Node],
    weights: np.ndarray,
    force: np.ndarray,
) -> dict[Node, np.ndarray]:
    return {
        node: (
            float(weight)
            * force
        )
        for node, weight
        in zip(nodes, weights)
    }


def merge_force_maps(
    *force_maps: dict[
        Node,
        np.ndarray,
    ],
) -> dict[Node, np.ndarray]:
    merged: dict[
        Node,
        np.ndarray,
    ] = {}

    for force_map in force_maps:
        for node, force in (
            force_map.items()
        ):
            if node not in merged:
                merged[node] = np.zeros(
                    3,
                    dtype=float,
                )

            merged[node] += force

    return merged


def resultant_force(
    force_map: dict[
        Node,
        np.ndarray,
    ],
) -> np.ndarray:
    result = np.zeros(
        3,
        dtype=float,
    )

    for force in force_map.values():
        result += force

    return result


def resultant_moment(
    force_map: dict[
        Node,
        np.ndarray,
    ],
    origin: np.ndarray,
) -> np.ndarray:
    result = np.zeros(
        3,
        dtype=float,
    )

    for node, force in (
        force_map.items()
    ):
        result += np.cross(
            node.position - origin,
            force,
        )

    return result


def path_kinematics(
    points: list[np.ndarray],
    velocities: list[np.ndarray],
) -> tuple[
    float,
    float,
    list[np.ndarray],
    list[float],
]:
    units: list[np.ndarray] = []
    lengths: list[float] = []

    total_length = 0.0
    total_length_velocity = 0.0

    for index in range(
        len(points) - 1
    ):
        displacement = (
            points[index + 1]
            - points[index]
        )

        length = float(
            np.linalg.norm(
                displacement
            )
        )

        if length <= 1e-12:
            raise RuntimeError(
                "Path segment collapsed"
            )

        unit = (
            displacement / length
        )

        relative_velocity = (
            velocities[index + 1]
            - velocities[index]
        )

        segment_length_velocity = (
            float(
                np.dot(
                    relative_velocity,
                    unit,
                )
            )
        )

        units.append(unit)
        lengths.append(length)

        total_length += length
        total_length_velocity += (
            segment_length_velocity
        )

    return (
        total_length,
        total_length_velocity,
        units,
        lengths,
    )


def tension_only_force(
    *,
    total_length: float,
    total_length_velocity: float,
    rest_length: float,
    stiffness: float,
    damping: float,
) -> float:
    extension = (
        total_length
        - rest_length
    )

    if extension <= 0.0:
        return 0.0

    tension = (
        stiffness * extension
        + damping
        * total_length_velocity
    )

    return max(
        0.0,
        float(tension),
    )


def path_point_forces(
    *,
    units: list[np.ndarray],
    tension: float,
) -> list[np.ndarray]:
    point_count = (
        len(units) + 1
    )

    forces = [
        np.zeros(
            3,
            dtype=float,
        )
        for _ in range(point_count)
    ]

    # First endpoint.
    forces[0] += (
        tension * units[0]
    )

    # Every interior point receives
    # the resultant of the two adjacent
    # segment tensions.
    for index in range(
        1,
        point_count - 1,
    ):
        forces[index] += (
            tension
            * (
                units[index]
                - units[index - 1]
            )
        )

    # Last endpoint.
    forces[-1] += (
        -tension
        * units[-1]
    )

    return forces


def point_force_resultant(
    forces: list[np.ndarray],
) -> np.ndarray:
    return sum(
        forces,
        start=np.zeros(
            3,
            dtype=float,
        ),
    )


def point_force_moment(
    points: list[np.ndarray],
    forces: list[np.ndarray],
    origin: np.ndarray,
) -> np.ndarray:
    result = np.zeros(
        3,
        dtype=float,
    )

    for point, force in zip(
        points,
        forces,
    ):
        result += np.cross(
            point - origin,
            force,
        )

    return result


def main() -> None:
    # =========================================================
    # Existing ROIF mechanics only.
    # =========================================================

    solver = Solver(
        iterations=100,
        tolerance=1e-10,
        correction_tolerance=1e-12,
        minimum_iterations=2,
        relaxation=1.0,
        enable_velocity_damping=False,
        detect_stagnation=False,
    )

    network = Network(
        gravity=None,
        global_damping=0.01,
        solver=solver,
        record_history=False,
    )

    # =========================================================
    # Three parent structures.
    #
    # A     = fixed origin structure
    # GUIDE = fixed intermediate structure
    # B     = free terminal structure
    #
    # "Structure" is deliberately generic.
    # No anatomical claim is made.
    # =========================================================

    structure_a = make_structure(
        prefix="STRUCTURE_A",
        origin=np.array(
            [0.0, 0.0, 0.0],
            dtype=float,
        ),
        fixed=True,
    )

    guide_structure = make_structure(
        prefix="GUIDE_STRUCTURE",
        origin=np.array(
            [0.14, 0.12, 0.03],
            dtype=float,
        ),
        fixed=True,
    )

    structure_b = make_structure(
        prefix="STRUCTURE_B",
        origin=np.array(
            [0.32, 0.02, 0.06],
            dtype=float,
        ),
        fixed=False,
    )

    all_nodes = (
        structure_a
        + guide_structure
        + structure_b
    )

    for node in all_nodes:
        network.add_node(node)

    add_shape_constraints(
        network,
        structure_a,
        "STRUCTURE_A",
    )

    add_shape_constraints(
        network,
        guide_structure,
        "GUIDE_STRUCTURE",
    )

    add_shape_constraints(
        network,
        structure_b,
        "STRUCTURE_B",
    )

    # =========================================================
    # Body-relative virtual attachments.
    #
    # No attachment is an ordinary Node.
    # =========================================================

    weights_a = np.array(
        [0.10, 0.70, 0.10, 0.10],
        dtype=float,
    )

    weights_guide = np.array(
        [0.25, 0.15, 0.50, 0.10],
        dtype=float,
    )

    weights_b = np.array(
        [0.60, 0.10, 0.20, 0.10],
        dtype=float,
    )

    for name, weights in (
        ("weights_a", weights_a),
        (
            "weights_guide",
            weights_guide,
        ),
        ("weights_b", weights_b),
    ):
        if not np.isclose(
            np.sum(weights),
            1.0,
        ):
            raise RuntimeError(
                f"{name} must sum to one"
            )

    parent_structures = [
        structure_a,
        guide_structure,
        structure_b,
    ]

    parent_weights = [
        weights_a,
        weights_guide,
        weights_b,
    ]

    # =========================================================
    # Initial path state.
    # =========================================================

    initial_points = [
        attachment_position(
            nodes,
            weights,
        )
        for nodes, weights
        in zip(
            parent_structures,
            parent_weights,
        )
    ]

    initial_velocities = [
        attachment_velocity(
            nodes,
            weights,
        )
        for nodes, weights
        in zip(
            parent_structures,
            parent_weights,
        )
    ]

    (
        initial_path_length,
        initial_path_velocity,
        initial_units,
        initial_segment_lengths,
    ) = path_kinematics(
        initial_points,
        initial_velocities,
    )

    rest_length = (
        initial_path_length
        * 0.85
    )

    stiffness = 160.0
    tissue_damping = 1.2

    initial_tension = (
        tension_only_force(
            total_length=(
                initial_path_length
            ),
            total_length_velocity=(
                initial_path_velocity
            ),
            rest_length=rest_length,
            stiffness=stiffness,
            damping=tissue_damping,
        )
    )

    initial_point_forces = (
        path_point_forces(
            units=initial_units,
            tension=initial_tension,
        )
    )

    # =========================================================
    # Static mechanics audit.
    #
    # One scalar tension generates all forces
    # along the complete path.
    # =========================================================

    point_resultant_error = float(
        np.linalg.norm(
            point_force_resultant(
                initial_point_forces
            )
        )
    )

    origin = np.zeros(
        3,
        dtype=float,
    )

    point_moment_error = float(
        np.linalg.norm(
            point_force_moment(
                initial_points,
                initial_point_forces,
                origin,
            )
        )
    )

    distributed_maps = [
        distribute_attachment_force(
            nodes,
            weights,
            force,
        )
        for nodes, weights, force
        in zip(
            parent_structures,
            parent_weights,
            initial_point_forces,
        )
    ]

    attachment_force_errors: list[
        float
    ] = []

    attachment_moment_errors: list[
        float
    ] = []

    for (
        nodes,
        point,
        point_force,
        distributed,
    ) in zip(
        parent_structures,
        initial_points,
        initial_point_forces,
        distributed_maps,
    ):
        attachment_force_errors.append(
            float(
                np.linalg.norm(
                    resultant_force(
                        distributed
                    )
                    - point_force
                )
            )
        )

        expected_moment = np.cross(
            point - origin,
            point_force,
        )

        distributed_moment = (
            resultant_moment(
                distributed,
                origin,
            )
        )

        attachment_moment_errors.append(
            float(
                np.linalg.norm(
                    distributed_moment
                    - expected_moment
                )
            )
        )

    initial_mechanical_power = float(
        sum(
            np.dot(force, velocity)
            for force, velocity
            in zip(
                initial_point_forces,
                initial_velocities,
            )
        )
    )

    power_identity_error = abs(
        initial_mechanical_power
        + initial_tension
        * initial_path_velocity
    )

    guide_force_magnitude = float(
        np.linalg.norm(
            initial_point_forces[1]
        )
    )

    # =========================================================
    # Dynamic proof.
    # =========================================================

    initial_centroid_b = centroid(
        structure_b
    )

    initial_axis_b = (
        structure_b[1].position
        - structure_b[0].position
    )
    initial_axis_b /= np.linalg.norm(
        initial_axis_b
    )

    initial_distances_b = (
        pair_distances(
            structure_b
        )
    )

    dt = 0.0005
    steps = 1200

    tension_history: list[float] = []
    path_length_history: list[
        float
    ] = []

    guide_force_history: list[
        float
    ] = []

    for _ in range(steps):
        points = [
            attachment_position(
                nodes,
                weights,
            )
            for nodes, weights
            in zip(
                parent_structures,
                parent_weights,
            )
        ]

        velocities = [
            attachment_velocity(
                nodes,
                weights,
            )
            for nodes, weights
            in zip(
                parent_structures,
                parent_weights,
            )
        ]

        (
            total_length,
            total_length_velocity,
            units,
            _,
        ) = path_kinematics(
            points,
            velocities,
        )

        tension = (
            tension_only_force(
                total_length=(
                    total_length
                ),
                total_length_velocity=(
                    total_length_velocity
                ),
                rest_length=rest_length,
                stiffness=stiffness,
                damping=tissue_damping,
            )
        )

        point_forces = (
            path_point_forces(
                units=units,
                tension=tension,
            )
        )

        force_maps = [
            distribute_attachment_force(
                nodes,
                weights,
                force,
            )
            for nodes, weights, force
            in zip(
                parent_structures,
                parent_weights,
                point_forces,
            )
        ]

        external_forces = (
            merge_force_maps(
                *force_maps
            )
        )

        network.step(
            dt=dt,
            external_forces=(
                external_forces
            ),
            update_materials=False,
            include_active=False,
            solve_constraints=True,
            record=False,
        )

        tension_history.append(
            tension
        )

        path_length_history.append(
            total_length
        )

        guide_force_history.append(
            float(
                np.linalg.norm(
                    point_forces[1]
                )
            )
        )

    # =========================================================
    # Final measurements.
    # =========================================================

    final_points = [
        attachment_position(
            nodes,
            weights,
        )
        for nodes, weights
        in zip(
            parent_structures,
            parent_weights,
        )
    ]

    final_velocities = [
        attachment_velocity(
            nodes,
            weights,
        )
        for nodes, weights
        in zip(
            parent_structures,
            parent_weights,
        )
    ]

    (
        final_path_length,
        final_path_velocity,
        final_units,
        final_segment_lengths,
    ) = path_kinematics(
        final_points,
        final_velocities,
    )

    final_tension = (
        tension_only_force(
            total_length=(
                final_path_length
            ),
            total_length_velocity=(
                final_path_velocity
            ),
            rest_length=rest_length,
            stiffness=stiffness,
            damping=tissue_damping,
        )
    )

    final_centroid_b = centroid(
        structure_b
    )

    final_axis_b = (
        structure_b[1].position
        - structure_b[0].position
    )
    final_axis_b /= np.linalg.norm(
        final_axis_b
    )

    final_distances_b = (
        pair_distances(
            structure_b
        )
    )

    distance_errors_b = {
        key: abs(
            final_distances_b[key]
            - initial_distances_b[key]
        )
        for key
        in initial_distances_b
    }

    maximum_distance_error_b = max(
        distance_errors_b.values()
    )

    centroid_displacement_b = float(
        np.linalg.norm(
            final_centroid_b
            - initial_centroid_b
        )
    )

    axis_dot = float(
        np.clip(
            np.dot(
                initial_axis_b,
                final_axis_b,
            ),
            -1.0,
            1.0,
        )
    )

    orientation_change_degrees = float(
        np.degrees(
            np.arccos(
                axis_dot
            )
        )
    )

    # =========================================================
    # Criteria.
    # =========================================================

    shape_preserved = bool(
        maximum_distance_error_b
        <= 1e-7
    )

    structure_b_moved = bool(
        centroid_displacement_b
        > 1e-6
    )

    structure_b_rotated = bool(
        orientation_change_degrees
        > 1e-3
    )

    path_resultant_exact = bool(
        point_resultant_error
        <= 1e-12
    )

    path_moment_exact = bool(
        point_moment_error
        <= 1e-12
    )

    attachment_force_transfer_exact = bool(
        max(
            attachment_force_errors
        )
        <= 1e-12
    )

    attachment_moment_transfer_exact = bool(
        max(
            attachment_moment_errors
        )
        <= 1e-12
    )

    virtual_work_consistent = bool(
        power_identity_error
        <= 1e-12
    )

    intermediate_force_present = bool(
        guide_force_magnitude
        > 1e-8
    )

    passive_tension_present = bool(
        initial_tension > 0.0
    )

    path_shortened = bool(
        final_path_length
        < initial_path_length
    )

    proof_passed = bool(
        shape_preserved
        and structure_b_moved
        and structure_b_rotated
        and path_resultant_exact
        and path_moment_exact
        and attachment_force_transfer_exact
        and attachment_moment_transfer_exact
        and virtual_work_consistent
        and intermediate_force_present
        and passive_tension_present
        and path_shortened
    )

    result = {
        "experiment": (
            "roif_multipoint_passive_path_proof"
        ),
        "purpose": (
            "Minimal proof that one passive "
            "tension element can act through "
            "a multipoint body-relative path "
            "with one total path length and "
            "one scalar tension."
        ),
        "new_mechanics_engine_added": False,
        "production_path_class_added": False,
        "production_attachment_class_added": False,
        "production_tissue_class_added": False,
        "ordinary_attachment_nodes_used": False,
        "wrapping_tested": False,
        "contact_tested": False,
        "anatomical_validity_tested": False,
        "path": {
            "point_count": 3,
            "segment_count": 2,
            "initial_segment_lengths": (
                initial_segment_lengths
            ),
            "final_segment_lengths": (
                final_segment_lengths
            ),
            "initial_total_length": (
                initial_path_length
            ),
            "rest_length": (
                rest_length
            ),
            "final_total_length": (
                final_path_length
            ),
            "single_scalar_tension": True,
        },
        "passive_tissue": {
            "model": (
                "minimal tension-only "
                "linear spring-damper "
                "over total path length"
            ),
            "stiffness": stiffness,
            "damping": tissue_damping,
            "initial_tension": (
                initial_tension
            ),
            "maximum_tension": float(
                max(tension_history)
            ),
            "final_tension": (
                final_tension
            ),
        },
        "intermediate_point": {
            "initial_force_magnitude": (
                guide_force_magnitude
            ),
            "maximum_force_magnitude": (
                float(
                    max(
                        guide_force_history
                    )
                )
            ),
        },
        "static_path_audit": {
            "resultant_force_error": (
                point_resultant_error
            ),
            "resultant_moment_error": (
                point_moment_error
            ),
            "maximum_attachment_force_error": (
                max(
                    attachment_force_errors
                )
            ),
            "maximum_attachment_moment_error": (
                max(
                    attachment_moment_errors
                )
            ),
            "virtual_work_error": (
                power_identity_error
            ),
        },
        "structure_b_motion": {
            "centroid_displacement": (
                centroid_displacement_b
            ),
            "orientation_change_degrees": (
                orientation_change_degrees
            ),
        },
        "structure_b_shape": {
            "maximum_pair_distance_error": (
                maximum_distance_error_b
            ),
            "pair_distance_errors": {
                f"{i}-{j}": error
                for (i, j), error
                in distance_errors_b.items()
            },
        },
        "solver": {
            "iterations_last_step": (
                solver.last_statistics
                .iterations
            ),
            "converged_last_step": (
                solver.last_statistics
                .converged
            ),
            "maximum_violation_last_step": (
                solver.last_statistics
                .maximum_violation
            ),
            "divergence_detected": (
                solver.last_statistics
                .divergence_detected
            ),
        },
        "criteria": {
            "shape_preserved": (
                shape_preserved
            ),
            "structure_b_moved": (
                structure_b_moved
            ),
            "structure_b_rotated": (
                structure_b_rotated
            ),
            "path_resultant_exact": (
                path_resultant_exact
            ),
            "path_moment_exact": (
                path_moment_exact
            ),
            "attachment_force_transfer_exact": (
                attachment_force_transfer_exact
            ),
            "attachment_moment_transfer_exact": (
                attachment_moment_transfer_exact
            ),
            "virtual_work_consistent": (
                virtual_work_consistent
            ),
            "intermediate_force_present": (
                intermediate_force_present
            ),
            "passive_tension_present": (
                passive_tension_present
            ),
            "path_shortened": (
                path_shortened
            ),
        },
        "proof_passed": proof_passed,
    }

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )

    if not proof_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
