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


def attachment_position(
    nodes: list[Node],
    weights: np.ndarray,
) -> np.ndarray:
    point = np.zeros(3, dtype=float)

    for node, weight in zip(
        nodes,
        weights,
    ):
        point += (
            float(weight)
            * node.position
        )

    return point


def attachment_velocity(
    nodes: list[Node],
    weights: np.ndarray,
) -> np.ndarray:
    velocity = np.zeros(3, dtype=float)

    for node, weight in zip(
        nodes,
        weights,
    ):
        velocity += (
            float(weight)
            * node.velocity
        )

    return velocity


def distribute_attachment_force(
    nodes: list[Node],
    weights: np.ndarray,
    force: np.ndarray,
) -> dict[Node, np.ndarray]:
    return {
        node: float(weight) * force
        for node, weight in zip(
            nodes,
            weights,
        )
    }


def merge_forces(
    *force_maps: dict[Node, np.ndarray],
) -> dict[Node, np.ndarray]:
    merged: dict[Node, np.ndarray] = {}

    for force_map in force_maps:
        for node, force in force_map.items():
            if node not in merged:
                merged[node] = np.zeros(
                    3,
                    dtype=float,
                )

            merged[node] += force

    return merged


def resultant_force(
    distributed: dict[Node, np.ndarray],
) -> np.ndarray:
    result = np.zeros(3, dtype=float)

    for force in distributed.values():
        result += force

    return result


def resultant_moment(
    distributed: dict[Node, np.ndarray],
    origin: np.ndarray,
) -> np.ndarray:
    result = np.zeros(3, dtype=float)

    for node, force in distributed.items():
        result += np.cross(
            node.position - origin,
            force,
        )

    return result


def make_bone_cluster(
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


def add_rigid_cluster_constraints(
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


def passive_tissue_force(
    *,
    point_a: np.ndarray,
    point_b: np.ndarray,
    velocity_a: np.ndarray,
    velocity_b: np.ndarray,
    rest_length: float,
    stiffness: float,
    damping: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    displacement = (
        point_b - point_a
    )

    length = float(
        np.linalg.norm(displacement)
    )

    if length <= 1e-12:
        zero = np.zeros(
            3,
            dtype=float,
        )
        return (
            zero,
            zero,
            0.0,
            length,
        )

    direction = (
        displacement / length
    )

    extension = (
        length - rest_length
    )

    if extension <= 0.0:
        zero = np.zeros(
            3,
            dtype=float,
        )
        return (
            zero,
            zero,
            0.0,
            length,
        )

    relative_velocity = float(
        np.dot(
            velocity_b
            - velocity_a,
            direction,
        )
    )

    tension = (
        stiffness * extension
        + damping * relative_velocity
    )

    tension = max(
        0.0,
        float(tension),
    )

    force_on_a = (
        tension * direction
    )

    force_on_b = (
        -tension * direction
    )

    return (
        force_on_a,
        force_on_b,
        tension,
        length,
    )


def main() -> None:
    # =========================================================
    # 1. Existing ROIF mechanics only
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
    # 2. Two shape-preserving structures
    #
    # A is fixed support only for this minimal proof.
    # B is completely free.
    # =========================================================

    bone_a = make_bone_cluster(
        prefix="STRUCTURE_A",
        origin=np.array(
            [0.0, 0.0, 0.0],
            dtype=float,
        ),
        fixed=True,
    )

    bone_b = make_bone_cluster(
        prefix="STRUCTURE_B",
        origin=np.array(
            [0.30, 0.06, 0.02],
            dtype=float,
        ),
        fixed=False,
    )

    for node in (
        bone_a + bone_b
    ):
        network.add_node(node)

    add_rigid_cluster_constraints(
        network,
        bone_a,
        "STRUCTURE_A",
    )

    add_rigid_cluster_constraints(
        network,
        bone_b,
        "STRUCTURE_B",
    )

    # =========================================================
    # 3. Attachments are NOT nodes.
    #
    # They are positions derived from their parent structure.
    #
    # The unequal weights deliberately place attachment B
    # away from the centre of mass so tissue tension should
    # generate both translation and rotation.
    # =========================================================

    weights_a = np.array(
        [0.10, 0.70, 0.10, 0.10],
        dtype=float,
    )

    weights_b = np.array(
        [0.65, 0.10, 0.20, 0.05],
        dtype=float,
    )

    if not np.isclose(
        np.sum(weights_a),
        1.0,
    ):
        raise RuntimeError(
            "weights_a must sum to one"
        )

    if not np.isclose(
        np.sum(weights_b),
        1.0,
    ):
        raise RuntimeError(
            "weights_b must sum to one"
        )

    initial_centroid_b = centroid(
        bone_b
    )

    initial_axis_b = (
        bone_b[1].position
        - bone_b[0].position
    )
    initial_axis_b /= np.linalg.norm(
        initial_axis_b
    )

    initial_distances_b = (
        pair_distances(
            bone_b
        )
    )

    attachment_a_initial = (
        attachment_position(
            bone_a,
            weights_a,
        )
    )

    attachment_b_initial = (
        attachment_position(
            bone_b,
            weights_b,
        )
    )

    initial_tissue_length = float(
        np.linalg.norm(
            attachment_b_initial
            - attachment_a_initial
        )
    )

    # Deliberate prestretch.
    rest_length = (
        initial_tissue_length
        * 0.80
    )

    stiffness = 180.0
    tissue_damping = 1.5

    # =========================================================
    # 4. Static force-transfer audit before integration
    # =========================================================

    (
        initial_force_a,
        initial_force_b,
        initial_tension,
        _,
    ) = passive_tissue_force(
        point_a=attachment_a_initial,
        point_b=attachment_b_initial,
        velocity_a=attachment_velocity(
            bone_a,
            weights_a,
        ),
        velocity_b=attachment_velocity(
            bone_b,
            weights_b,
        ),
        rest_length=rest_length,
        stiffness=stiffness,
        damping=tissue_damping,
    )

    distributed_a = (
        distribute_attachment_force(
            bone_a,
            weights_a,
            initial_force_a,
        )
    )

    distributed_b = (
        distribute_attachment_force(
            bone_b,
            weights_b,
            initial_force_b,
        )
    )

    force_error_a = float(
        np.linalg.norm(
            resultant_force(
                distributed_a
            )
            - initial_force_a
        )
    )

    force_error_b = float(
        np.linalg.norm(
            resultant_force(
                distributed_b
            )
            - initial_force_b
        )
    )

    action_reaction_error = float(
        np.linalg.norm(
            initial_force_a
            + initial_force_b
        )
    )

    centroid_a_initial = centroid(
        bone_a
    )

    expected_moment_a = np.cross(
        attachment_a_initial
        - centroid_a_initial,
        initial_force_a,
    )

    expected_moment_b = np.cross(
        attachment_b_initial
        - initial_centroid_b,
        initial_force_b,
    )

    transferred_moment_a = (
        resultant_moment(
            distributed_a,
            centroid_a_initial,
        )
    )

    transferred_moment_b = (
        resultant_moment(
            distributed_b,
            initial_centroid_b,
        )
    )

    moment_error_a = float(
        np.linalg.norm(
            transferred_moment_a
            - expected_moment_a
        )
    )

    moment_error_b = float(
        np.linalg.norm(
            transferred_moment_b
            - expected_moment_b
        )
    )

    # =========================================================
    # 5. Dynamic tissue transmission
    # =========================================================

    dt = 0.0005
    steps = 1200

    tension_history: list[float] = []
    length_history: list[float] = []

    for _ in range(steps):
        point_a = attachment_position(
            bone_a,
            weights_a,
        )

        point_b = attachment_position(
            bone_b,
            weights_b,
        )

        velocity_a = (
            attachment_velocity(
                bone_a,
                weights_a,
            )
        )

        velocity_b = (
            attachment_velocity(
                bone_b,
                weights_b,
            )
        )

        (
            force_a,
            force_b,
            tension,
            tissue_length,
        ) = passive_tissue_force(
            point_a=point_a,
            point_b=point_b,
            velocity_a=velocity_a,
            velocity_b=velocity_b,
            rest_length=rest_length,
            stiffness=stiffness,
            damping=tissue_damping,
        )

        external_forces = merge_forces(
            distribute_attachment_force(
                bone_a,
                weights_a,
                force_a,
            ),
            distribute_attachment_force(
                bone_b,
                weights_b,
                force_b,
            ),
        )

        network.step(
            dt=dt,
            external_forces=external_forces,
            update_materials=False,
            include_active=False,
            solve_constraints=True,
            record=False,
        )

        tension_history.append(
            tension
        )

        length_history.append(
            tissue_length
        )

    # =========================================================
    # 6. Final measurements
    # =========================================================

    final_centroid_b = centroid(
        bone_b
    )

    final_axis_b = (
        bone_b[1].position
        - bone_b[0].position
    )
    final_axis_b /= np.linalg.norm(
        final_axis_b
    )

    final_distances_b = (
        pair_distances(
            bone_b
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
            np.arccos(axis_dot)
        )
    )

    attachment_b_final = (
        attachment_position(
            bone_b,
            weights_b,
        )
    )

    final_tissue_length = float(
        np.linalg.norm(
            attachment_b_final
            - attachment_position(
                bone_a,
                weights_a,
            )
        )
    )

    # Directional evidence that B moved toward A attachment.
    initial_vector_to_a = (
        attachment_a_initial
        - initial_centroid_b
    )

    centroid_motion = (
        final_centroid_b
        - initial_centroid_b
    )

    moved_toward_attachment = bool(
        np.dot(
            centroid_motion,
            initial_vector_to_a,
        )
        > 0.0
    )

    # =========================================================
    # 7. Proof criteria
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

    force_transfer_exact = bool(
        force_error_a <= 1e-12
        and force_error_b <= 1e-12
    )

    moment_transfer_exact = bool(
        moment_error_a <= 1e-12
        and moment_error_b <= 1e-12
    )

    action_reaction_exact = bool(
        action_reaction_error
        <= 1e-12
    )

    passive_tension_present = bool(
        initial_tension > 0.0
    )

    tissue_shortened = bool(
        final_tissue_length
        < initial_tissue_length
    )

    proof_passed = bool(
        shape_preserved
        and structure_b_moved
        and structure_b_rotated
        and moved_toward_attachment
        and force_transfer_exact
        and moment_transfer_exact
        and action_reaction_exact
        and passive_tension_present
        and tissue_shortened
    )

    result = {
        "experiment": (
            "roif_passive_attachment_transfer_proof"
        ),
        "purpose": (
            "Minimal proof of passive force transfer "
            "between two ROIF structures through "
            "body-relative virtual attachments."
        ),
        "new_mechanics_engine_added": False,
        "production_attachment_class_added": False,
        "production_tissue_class_added": False,
        "ordinary_attachment_nodes_used": False,
        "multipoint_path_tested": False,
        "anatomical_validity_tested": False,
        "structures": {
            "structure_a": (
                "fixed shape-preserving "
                "tetrahedral cluster"
            ),
            "structure_b": (
                "free shape-preserving "
                "tetrahedral cluster"
            ),
            "constraints_total": len(
                network.constraints
            ),
        },
        "passive_tissue": {
            "model": (
                "minimal tension-only "
                "linear spring-damper"
            ),
            "initial_length": (
                initial_tissue_length
            ),
            "rest_length": rest_length,
            "final_length": (
                final_tissue_length
            ),
            "stiffness": stiffness,
            "damping": tissue_damping,
            "initial_tension": (
                initial_tension
            ),
            "maximum_tension": float(
                max(tension_history)
            ),
            "final_tension": float(
                tension_history[-1]
            ),
        },
        "attachment_transfer": {
            "force_error_a": (
                force_error_a
            ),
            "force_error_b": (
                force_error_b
            ),
            "action_reaction_error": (
                action_reaction_error
            ),
            "moment_error_a": (
                moment_error_a
            ),
            "moment_error_b": (
                moment_error_b
            ),
        },
        "structure_b_motion": {
            "centroid_displacement": (
                centroid_displacement_b
            ),
            "orientation_change_degrees": (
                orientation_change_degrees
            ),
            "moved_toward_attachment": (
                moved_toward_attachment
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
                solver.last_statistics.iterations
            ),
            "converged_last_step": (
                solver.last_statistics.converged
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
            "moved_toward_attachment": (
                moved_toward_attachment
            ),
            "force_transfer_exact": (
                force_transfer_exact
            ),
            "moment_transfer_exact": (
                moment_transfer_exact
            ),
            "action_reaction_exact": (
                action_reaction_exact
            ),
            "passive_tension_present": (
                passive_tension_present
            ),
            "tissue_shortened": (
                tissue_shortened
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
