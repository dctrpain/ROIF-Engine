from __future__ import annotations

import json
from itertools import combinations

import numpy as np

from core.distance_constraint import DistanceConstraint
from core.network import Network
from core.node import Node
from core.solver import Solver


def centroid(nodes: list[Node]) -> np.ndarray:
    masses = np.asarray([node.mass for node in nodes], dtype=float)
    positions = np.vstack([node.position for node in nodes])
    return np.sum(positions * masses[:, None], axis=0) / np.sum(masses)


def pair_distances(nodes: list[Node]) -> dict[tuple[int, int], float]:
    result: dict[tuple[int, int], float] = {}

    for i, j in combinations(range(len(nodes)), 2):
        result[(i, j)] = float(
            np.linalg.norm(nodes[j].position - nodes[i].position)
        )

    return result


def attachment_position(
    nodes: list[Node],
    weights: np.ndarray,
) -> np.ndarray:
    return sum(
        (
            float(weight) * node.position
            for node, weight in zip(nodes, weights)
        ),
        start=np.zeros(3, dtype=float),
    )


def distribute_attachment_force(
    nodes: list[Node],
    weights: np.ndarray,
    force: np.ndarray,
) -> dict[Node, np.ndarray]:
    return {
        node: float(weight) * force
        for node, weight in zip(nodes, weights)
    }


def total_force(
    distributed: dict[Node, np.ndarray],
) -> np.ndarray:
    return sum(
        distributed.values(),
        start=np.zeros(3, dtype=float),
    )


def total_moment_about_origin(
    distributed: dict[Node, np.ndarray],
) -> np.ndarray:
    moment = np.zeros(3, dtype=float)

    for node, force in distributed.items():
        moment += np.cross(node.position, force)

    return moment


def main() -> None:
    # ---------------------------------------------------------
    # 1. Minimal non-coplanar "bone" cluster
    # ---------------------------------------------------------

    initial_positions = (
        np.array([0.0, 0.0, 0.0], dtype=float),
        np.array([0.12, 0.0, 0.0], dtype=float),
        np.array([0.0, 0.04, 0.0], dtype=float),
        np.array([0.0, 0.0, 0.03], dtype=float),
    )

    nodes = [
        Node(
            position=position,
            mass=0.25,
            node_id=f"BONE_{index}",
        )
        for index, position in enumerate(initial_positions)
    ]

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
        global_damping=0.0,
        solver=solver,
        record_history=False,
    )

    for node in nodes:
        network.add_node(node)

    # ---------------------------------------------------------
    # 2. Six equality constraints preserve tetrahedron shape
    # ---------------------------------------------------------

    for i, j in combinations(range(4), 2):
        network.add_constraint(
            DistanceConstraint(
                name=f"BONE_DISTANCE_{i}_{j}",
                node_a=nodes[i],
                node_b=nodes[j],
                target_length=float(
                    np.linalg.norm(
                        nodes[j].position - nodes[i].position
                    )
                ),
                mode="equality",
                compliance=0.0,
                damping=0.0,
            )
        )

    initial_distances = pair_distances(nodes)
    initial_centroid = centroid(nodes)

    initial_axis = (
        nodes[1].position - nodes[0].position
    )
    initial_axis /= np.linalg.norm(initial_axis)

    # ---------------------------------------------------------
    # 3. Virtual body-relative attachment
    #
    # P = sum(w_i * x_i), sum(w_i) = 1
    #
    # Asymmetric weights deliberately place the attachment away
    # from the centre of mass.
    # ---------------------------------------------------------

    weights = np.array(
        [0.10, 0.60, 0.20, 0.10],
        dtype=float,
    )

    if not np.isclose(np.sum(weights), 1.0):
        raise RuntimeError("Attachment weights must sum to one.")

    attachment_initial = attachment_position(
        nodes,
        weights,
    )

    applied_force = np.array(
        [0.0, 8.0, 3.0],
        dtype=float,
    )

    distributed_initial = distribute_attachment_force(
        nodes,
        weights,
        applied_force,
    )

    force_reconstruction_error = float(
        np.linalg.norm(
            total_force(distributed_initial)
            - applied_force
        )
    )

    expected_moment = np.cross(
        attachment_initial,
        applied_force,
    )

    distributed_moment = total_moment_about_origin(
        distributed_initial
    )

    moment_reconstruction_error = float(
        np.linalg.norm(
            distributed_moment - expected_moment
        )
    )

    # ---------------------------------------------------------
    # 4. Apply attachment force for a short interval
    # ---------------------------------------------------------

    dt = 0.0005
    force_steps = 40
    free_steps = 160

    for _ in range(force_steps):
        current_external_forces = (
            distribute_attachment_force(
                nodes,
                weights,
                applied_force,
            )
        )

        network.step(
            dt=dt,
            external_forces=current_external_forces,
            update_materials=False,
            include_active=False,
            solve_constraints=True,
            record=False,
        )

    # Then let the cluster continue freely.
    for _ in range(free_steps):
        network.step(
            dt=dt,
            external_forces=None,
            update_materials=False,
            include_active=False,
            solve_constraints=True,
            record=False,
        )

    # ---------------------------------------------------------
    # 5. Measurements
    # ---------------------------------------------------------

    final_distances = pair_distances(nodes)
    final_centroid = centroid(nodes)

    final_axis = (
        nodes[1].position - nodes[0].position
    )
    final_axis /= np.linalg.norm(final_axis)

    distance_errors = {
        key: abs(
            final_distances[key]
            - initial_distances[key]
        )
        for key in initial_distances
    }

    maximum_distance_error = max(
        distance_errors.values()
    )

    centroid_displacement = float(
        np.linalg.norm(
            final_centroid - initial_centroid
        )
    )

    axis_dot = float(
        np.clip(
            np.dot(initial_axis, final_axis),
            -1.0,
            1.0,
        )
    )

    orientation_change_degrees = float(
        np.degrees(
            np.arccos(axis_dot)
        )
    )

    attachment_final = attachment_position(
        nodes,
        weights,
    )

    # ---------------------------------------------------------
    # 6. Explicit proof criteria
    # ---------------------------------------------------------

    shape_preserved = bool(
        maximum_distance_error <= 1e-7
    )

    translated = bool(
        centroid_displacement > 1e-6
    )

    rotated = bool(
        orientation_change_degrees > 1e-3
    )

    force_transfer_exact = bool(
        force_reconstruction_error <= 1e-12
    )

    moment_transfer_exact = bool(
        moment_reconstruction_error <= 1e-12
    )

    proof_passed = bool(
        shape_preserved
        and translated
        and rotated
        and force_transfer_exact
        and moment_transfer_exact
    )

    result = {
        "experiment": "roif_rigid_bone_attachment_proof",
        "purpose": (
            "Minimal proof that existing ROIF primitives can "
            "represent a shape-preserving 3D bone cluster and "
            "receive force through a virtual body-relative "
            "attachment point."
        ),
        "new_mechanics_engine_added": False,
        "rigid_body_solver_added": False,
        "path_element_tested": False,
        "anatomical_validity_tested": False,
        "cluster": {
            "node_count": len(nodes),
            "distance_constraint_count": len(
                network.constraints
            ),
            "constraint_mode": "equality",
            "constraint_compliance": 0.0,
        },
        "attachment": {
            "weights": weights.tolist(),
            "initial_position": (
                attachment_initial.tolist()
            ),
            "final_position": (
                attachment_final.tolist()
            ),
            "applied_force": applied_force.tolist(),
            "force_reconstruction_error": (
                force_reconstruction_error
            ),
            "moment_reconstruction_error": (
                moment_reconstruction_error
            ),
        },
        "motion": {
            "initial_centroid": (
                initial_centroid.tolist()
            ),
            "final_centroid": (
                final_centroid.tolist()
            ),
            "centroid_displacement": (
                centroid_displacement
            ),
            "orientation_change_degrees": (
                orientation_change_degrees
            ),
        },
        "shape": {
            "maximum_pair_distance_error": (
                maximum_distance_error
            ),
            "pair_distance_errors": {
                f"{i}-{j}": error
                for (i, j), error
                in distance_errors.items()
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
                solver.last_statistics.maximum_violation
            ),
            "divergence_detected": (
                solver.last_statistics
                .divergence_detected
            ),
        },
        "criteria": {
            "shape_preserved": shape_preserved,
            "translated": translated,
            "rotated": rotated,
            "force_transfer_exact": (
                force_transfer_exact
            ),
            "moment_transfer_exact": (
                moment_transfer_exact
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
