import numpy as np
import pytest

from roif.worlds.tensegrity_3d_world import (
    Tensegrity3DWorld,
)


def test_world_contains_six_nodes():
    world = Tensegrity3DWorld()

    assert world.node_count == 6


def test_world_contains_three_struts_and_nine_cables():
    world = Tensegrity3DWorld()

    kinds = [member.kind for member in world.members]

    assert kinds.count("strut") == 3
    assert kinds.count("cable") == 9


def test_initial_structure_is_genuinely_prestressed():
    world = Tensegrity3DWorld()

    forces = world.member_axial_forces()

    strut_forces = forces[:3]
    cable_forces = forces[3:]

    assert np.all(strut_forces < 0.0)
    assert np.all(cable_forces > 0.0)


def test_initial_prestress_is_self_equilibrated():
    world = Tensegrity3DWorld()

    nodal_forces = world.nodal_internal_forces()

    assert np.max(
        np.linalg.norm(
            nodal_forces,
            axis=1,
        )
    ) < 1e-8


def test_initial_body_is_not_unstressed():
    world = Tensegrity3DWorld()

    forces = world.member_axial_forces()

    assert np.linalg.norm(forces) > 0.0


def test_equilibrium_remains_stable_without_external_input():
    world = Tensegrity3DWorld()

    initial_positions = world.positions.copy()

    for _ in range(1000):
        world.step(0.001)

    displacement = np.linalg.norm(
        world.positions - initial_positions,
        axis=1,
    )

    assert np.max(displacement) < 1e-10


def test_world_advances_time():
    world = Tensegrity3DWorld()

    world.step(0.01)

    assert world.time == pytest.approx(0.01)


def test_body_observation_contains_member_lengths():
    world = Tensegrity3DWorld()

    observation = world.body(
        "tensegrity_body_0"
    ).observe()

    assert "member_0_length" in observation.channels


def test_body_observation_contains_axial_force_sensation():
    world = Tensegrity3DWorld()

    observation = world.body(
        "tensegrity_body_0"
    ).observe()

    assert (
        "member_0_axial_force"
        in observation.channels
    )


def test_body_observation_does_not_expose_world_coordinates():
    world = Tensegrity3DWorld()

    observation = world.body(
        "tensegrity_body_0"
    ).observe()

    forbidden = {
        "x",
        "y",
        "z",
        "world_x",
        "world_y",
        "world_z",
    }

    assert forbidden.isdisjoint(
        observation.channels
    )


def test_world_ground_truth_contains_3d_coordinates():
    world = Tensegrity3DWorld()

    state = world.world_state()

    assert "node_0_x" in state.values
    assert "node_0_y" in state.values
    assert "node_0_z" in state.values


def test_ground_truth_and_experience_have_different_information():
    world = Tensegrity3DWorld()

    state = world.world_state()

    observation = world.body(
        "tensegrity_body_0"
    ).observe()

    assert (
        "node_0_x" in state.values
        and "node_0_x" not in observation.channels
    )


def test_world_has_no_reward():
    world = Tensegrity3DWorld()

    assert not hasattr(world, "reward")


def test_world_has_no_goal():
    world = Tensegrity3DWorld()

    assert not hasattr(world, "goal")