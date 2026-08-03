r"""
ROIF Engine Example 24

Dynamic 3D upper-limb model driven by a real distributed external force.

This example imports the passive tensegrity-inspired arm constructed in
Example 23, removes all initial velocity, and applies a time-dependent force
to a contact patch formed by the distal palm and finger nodes.

The motion is therefore caused by a real nodal external force passed through:

    Simulation.external_forces
        -> Network.step(...)
        -> Node.apply_force(...)
        -> acceleration = force / mass
        -> velocity integration
        -> position integration

Force scenario
--------------
0.00–0.35 s
    Quiet prestressed state.

0.35–1.10 s
    A smooth distributed force pushes the hand laterally and upward.
    This produces a clearly visible arm swing.

1.10–1.45 s
    Free mechanical response.

1.45–2.15 s
    A smaller opposite force pushes the hand back.

2.15–3.50 s
    Free oscillation and stabilization.

Important
---------
This is a simplified mechanical analogy, not an anatomically exact model.

The contact is represented as a distributed nodal force over the palm and
finger nodes. It is not yet a geometric collision with a plate or sphere.

Run:

    python examples\example_24_3d_arm_external_force.py

Geometry playback:

    python play.py output\example_24_3d_arm_external_force.roifrec \
        --mode geometry --interval 12 --repeat

Force playback:

    python play.py output\example_24_3d_arm_external_force.roifrec \
        --mode force --interval 12 --repeat
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sin
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from core.network import Network
from core.node import Node
from core.recorder import SimulationRecorder
from core.simulation import Simulation
from core.storage import RecordingStorage
from examples.example_23_3d_arm_tensegrity import (
    build_network as build_passive_arm,
)


OUTPUT_PATH = (
    PROJECT_ROOT
    / "output"
    / "example_24_3d_arm_external_force.roifrec"
)


# ---------------------------------------------------------------------
# Demonstration parameters
# ---------------------------------------------------------------------

SIMULATION_DT = 0.0002
SIMULATION_DURATION = 2.8
SIMULATION_STEPS = int(
    SIMULATION_DURATION / SIMULATION_DT
)
SAMPLE_EVERY = 40

# First pulse: strong visible arm swing.
PRIMARY_FORCE_PEAK_N = 34.0
PRIMARY_START_S = 0.35
PRIMARY_END_S = 1.10

# Second pulse: smaller opposite movement.
RETURN_FORCE_PEAK_N = 22.0
RETURN_START_S = 1.45
RETURN_END_S = 2.15

# Direction of the first force.
#
# +Y produces lateral movement.
# +Z lifts the hand.
PRIMARY_DIRECTION = np.asarray(
    [0.0, 0.91, 0.42],
    dtype=float,
)

# Direction of the return force.
RETURN_DIRECTION = np.asarray(
    [0.0, -0.94, 0.34],
    dtype=float,
)


def normalized(
    vector: np.ndarray,
) -> np.ndarray:
    magnitude = float(
        np.linalg.norm(vector)
    )

    if magnitude <= 0.0:
        raise ValueError(
            "force direction cannot be zero"
        )

    return vector / magnitude


PRIMARY_DIRECTION = normalized(
    PRIMARY_DIRECTION
)
RETURN_DIRECTION = normalized(
    RETURN_DIRECTION
)


@dataclass(frozen=True, slots=True)
class ContactNode:
    """
    One node in the distributed hand contact patch.

    weight:
        Fraction of the total external force received by the node.
    """

    node: Node
    weight: float


class DistributedHandForce:
    """
    Time-dependent external-force provider for Simulation.

    The provider returns a mapping:

        {Node: force_vector}

    at every physical step.

    It also stores the current force state on the Network. RecordingFrame
    deep-copies the Network, so these values become available to future
    Viewer overlays without changing the recording format.
    """

    VERSION = "1.0"

    def __init__(
        self,
        network: Network,
    ) -> None:
        self.network = network

        self.contact_nodes = (
            ContactNode(
                self._node("P_DF"),
                0.18,
            ),
            ContactNode(
                self._node("P_DB"),
                0.18,
            ),
            ContactNode(
                self._node("F_FRONT"),
                0.20,
            ),
            ContactNode(
                self._node("F_MIDDLE"),
                0.24,
            ),
            ContactNode(
                self._node("F_BACK"),
                0.20,
            ),
        )

        total_weight = sum(
            item.weight
            for item in self.contact_nodes
        )

        if not np.isclose(
            total_weight,
            1.0,
        ):
            raise ValueError(
                "contact weights must sum to 1.0"
            )

        self.current_phase = "quiet"
        self.current_force = np.zeros(
            3,
            dtype=float,
        )
        self.current_magnitude = 0.0

        self._write_network_state()

    def _node(
        self,
        node_id: str,
    ) -> Node:
        for node in self.network.nodes:
            if str(node.id) == node_id:
                return node

        raise KeyError(
            f"required contact node not found: {node_id}"
        )

    @staticmethod
    def smooth_pulse(
        time: float,
        *,
        start: float,
        end: float,
        peak: float,
    ) -> float:
        """
        Half-sine force pulse.

        The force is zero at start and end and reaches peak at the middle:

            F(t) = F_peak sin(pi tau)

        where tau is normalized to [0, 1].
        """

        if time < start or time > end:
            return 0.0

        duration = end - start

        if duration <= 0.0:
            raise ValueError(
                "pulse end must be greater than start"
            )

        tau = (
            time - start
        ) / duration

        return float(
            peak * sin(pi * tau)
        )

    def force_at_time(
        self,
        time: float,
    ) -> tuple[str, np.ndarray]:
        primary_magnitude = self.smooth_pulse(
            time,
            start=PRIMARY_START_S,
            end=PRIMARY_END_S,
            peak=PRIMARY_FORCE_PEAK_N,
        )

        if primary_magnitude > 0.0:
            return (
                "primary_hand_push",
                PRIMARY_DIRECTION
                * primary_magnitude,
            )

        return_magnitude = self.smooth_pulse(
            time,
            start=RETURN_START_S,
            end=RETURN_END_S,
            peak=RETURN_FORCE_PEAK_N,
        )

        if return_magnitude > 0.0:
            return (
                "return_hand_push",
                RETURN_DIRECTION
                * return_magnitude,
            )

        if time < PRIMARY_START_S:
            phase = "quiet_prestressed_state"
        elif time < RETURN_START_S:
            phase = "free_response_after_primary_force"
        elif time < RETURN_END_S:
            phase = "return_hand_push"
        else:
            phase = "free_oscillation_and_stabilization"

        return (
            phase,
            np.zeros(
                3,
                dtype=float,
            ),
        )

    def _write_network_state(
        self,
    ) -> None:
        """
        Store force information on the network for recording/viewer use.

        These attributes do not affect the physical solver.
        """

        self.network.external_force_phase = (
            self.current_phase
        )
        self.network.external_force_vector = (
            self.current_force.copy()
        )
        self.network.external_force_magnitude = float(
            self.current_magnitude
        )
        self.network.external_force_units = "N"
        self.network.external_force_application = (
            "distributed hand contact patch"
        )
        self.network.external_force_contact_nodes = [
            str(item.node.id)
            for item in self.contact_nodes
        ]
        self.network.external_force_contact_weights = {
            str(item.node.id): float(item.weight)
            for item in self.contact_nodes
        }

    def __call__(
        self,
        simulation: Simulation,
    ) -> dict[Node, np.ndarray] | None:
        phase, total_force = self.force_at_time(
            simulation.time
        )

        magnitude = float(
            np.linalg.norm(total_force)
        )

        self.current_phase = phase
        self.current_force = (
            total_force.copy()
        )
        self.current_magnitude = magnitude

        self._write_network_state()

        if magnitude <= 0.0:
            return None

        distributed: dict[
            Node,
            np.ndarray,
        ] = {}

        for contact in self.contact_nodes:
            distributed[contact.node] = (
                total_force
                * contact.weight
            )

        return distributed


def prepare_network() -> Network:
    """
    Build the Example 23 arm and remove all initial excitation.

    The only cause of motion in Example 24 is the external-force provider.
    """

    network = build_passive_arm()

    # Less damping than Example 23 so a full arm swing remains visible.
    network.global_damping = 0.42

    for node in network.nodes:
        if not node.fixed:
            node.zero_velocity()
            node.clear_force()

    return network


def node_by_id(
    network: Network,
    node_id: str,
) -> Node:
    for node in network.nodes:
        if str(node.id) == node_id:
            return node

    raise KeyError(
        f"unknown node id: {node_id}"
    )


def print_force_scenario(
    controller: DistributedHandForce,
) -> None:
    print()
    print("External force scenario")
    print("-" * 96)
    print(
        f"Primary peak  : "
        f"{PRIMARY_FORCE_PEAK_N:.1f} N"
    )
    print(
        f"Primary time  : "
        f"{PRIMARY_START_S:.2f}–"
        f"{PRIMARY_END_S:.2f} s"
    )
    print(
        "Primary vector: "
        f"[{PRIMARY_DIRECTION[0]:+.3f}, "
        f"{PRIMARY_DIRECTION[1]:+.3f}, "
        f"{PRIMARY_DIRECTION[2]:+.3f}]"
    )
    print(
        f"Return peak   : "
        f"{RETURN_FORCE_PEAK_N:.1f} N"
    )
    print(
        f"Return time   : "
        f"{RETURN_START_S:.2f}–"
        f"{RETURN_END_S:.2f} s"
    )
    print(
        "Return vector : "
        f"[{RETURN_DIRECTION[0]:+.3f}, "
        f"{RETURN_DIRECTION[1]:+.3f}, "
        f"{RETURN_DIRECTION[2]:+.3f}]"
    )

    print()
    print("Distributed contact patch")
    print("-" * 96)

    for contact in controller.contact_nodes:
        print(
            f"{str(contact.node.id):12s} "
            f"weight={contact.weight:5.2f} "
            f"primary_peak_share="
            f"{PRIMARY_FORCE_PEAK_N * contact.weight:6.2f} N"
        )


def print_initial_state(
    network: Network,
) -> None:
    print()
    print("Initial dynamic state")
    print("-" * 96)

    moving_nodes = [
        node
        for node in network.nodes
        if not node.fixed
    ]

    maximum_speed = max(
        node.speed()
        for node in moving_nodes
    )

    print(
        f"Maximum initial speed : "
        f"{maximum_speed:.9f}"
    )
    print(
        "Expected value        : 0.000000000"
    )
    print(
        "Cause of motion       : external force only"
    )


def print_final_state(
    network: Network,
) -> None:
    print()
    print("Final key-node positions")
    print("-" * 96)

    key_ids = (
        "H_PF",
        "H_DF",
        "R_D",
        "U_D",
        "P_DF",
        "P_DB",
        "F_FRONT",
        "F_MIDDLE",
        "F_BACK",
    )

    for node_id in key_ids:
        node = node_by_id(
            network,
            node_id,
        )

        coordinates = ", ".join(
            f"{float(value):+.6f}"
            for value in node.position
        )

        print(
            f"{node_id:10s} "
            f"position=[{coordinates}] "
            f"speed={node.speed():.5f}"
        )


def main() -> None:
    print("=" * 96)
    print("ROIF Engine - Example 24")
    print("3D upper-limb response to a real distributed external force")
    print("=" * 96)

    network = prepare_network()
    controller = DistributedHandForce(
        network
    )

    print(
        f"Nodes          : {len(network.nodes)}"
    )
    print(
        f"Elements       : {len(network.elements)}"
    )
    print(
        f"Physical time  : "
        f"{SIMULATION_DURATION:.2f} s"
    )
    print(
        f"Time step      : "
        f"{SIMULATION_DT:.7f} s"
    )
    print(
        f"Physical steps : "
        f"{SIMULATION_STEPS}"
    )

    print_initial_state(network)
    print_force_scenario(controller)

    simulation = Simulation(
        network,
        dt=SIMULATION_DT,
        external_forces=controller,
        update_materials=False,
        include_active=False,
        solve_constraints=True,
        record_network_history=False,
        record_simulation_history=False,
    )

    print()
    print("Running force-driven physical simulation...")
    print("-" * 96)

    recording = SimulationRecorder(
        simulation
    ).run_steps(
        SIMULATION_STEPS,
        sample_every=SAMPLE_EVERY,
        include_initial=True,
        include_final=True,
        metadata={
            "name": (
                "3D arm driven by distributed external force"
            ),
            "purpose": (
                "visible force-to-motion causal demonstration"
            ),
            "model_scope": (
                "simplified shoulder, arm, forearm, palm, fingers"
            ),
            "force_model": (
                "time-dependent distributed nodal force"
            ),
            "force_units": "N",
            "primary_force_peak_n": (
                PRIMARY_FORCE_PEAK_N
            ),
            "primary_force_start_s": (
                PRIMARY_START_S
            ),
            "primary_force_end_s": (
                PRIMARY_END_S
            ),
            "primary_force_direction": (
                PRIMARY_DIRECTION.tolist()
            ),
            "return_force_peak_n": (
                RETURN_FORCE_PEAK_N
            ),
            "return_force_start_s": (
                RETURN_START_S
            ),
            "return_force_end_s": (
                RETURN_END_S
            ),
            "return_force_direction": (
                RETURN_DIRECTION.tolist()
            ),
            "contact_nodes": [
                str(item.node.id)
                for item in controller.contact_nodes
            ],
            "contact_weights": {
                str(item.node.id): float(item.weight)
                for item in controller.contact_nodes
            },
            "initial_velocity": 0.0,
            "active_muscles": False,
            "anatomical_accuracy": False,
            "physical_duration_seconds": (
                SIMULATION_DURATION
            ),
            "requested_steps": (
                SIMULATION_STEPS
            ),
            "sample_every": SAMPLE_EVERY,
        },
    )

    saved_path = RecordingStorage.save(
        recording,
        OUTPUT_PATH,
        compress=True,
        overwrite=True,
    )

    info = RecordingStorage.inspect(
        saved_path,
        verify_checksum=True,
    )

    print(f"Physical steps : {info.physical_steps}")
    print(f"Frames         : {info.frame_count}")
    print(f"Duration       : {info.duration:.6f} s")
    print(f"File           : {saved_path}")
    print(
        f"File size      : "
        f"{saved_path.stat().st_size:,} bytes"
    )
    print(f"SHA-256        : {info.payload_sha256}")

    print_final_state(
        simulation.network
    )

    print()
    print("Recording created successfully.")
    print()
    print("What physically caused the movement:")
    print(
        f"  A {PRIMARY_FORCE_PEAK_N:.1f} N "
        "distributed hand force,"
    )
    print(
        "  applied to P_DF, P_DB and three finger nodes,"
    )
    print(
        "  followed by a smaller force in the "
        "opposite lateral direction."
    )

    print()
    print("Geometry playback:")
    print(
        "python play.py "
        "output\\example_24_3d_arm_external_force.roifrec "
        "--mode geometry --interval 12 --repeat"
    )

    print()
    print("Force playback:")
    print(
        "python play.py "
        "output\\example_24_3d_arm_external_force.roifrec "
        "--mode force --interval 12 --repeat"
    )

    print()
    print(
        "The current Player shows the physical response and "
        "element-force field."
    )
    print(
        "The explicit red force arrow and live F(t) label "
        "will require the next Viewer overlay step."
    )


if __name__ == "__main__":
    main()