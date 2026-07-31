from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import pytest

from core.network import Network
from core.node import Node
from core.recorder import SimulationRecorder
from core.simulation import Simulation
from core.storage import RecordingStorage
from play import (
    build_parser,
    main,
    run_player,
    validate_arguments,
)


def make_recording_file(
    tmp_path: Path,
) -> Path:
    network = Network(
        gravity=[0.0, -10.0],
        record_history=False,
    )
    network.add_node(
        Node(
            position=[0.0, 0.0],
            velocity=[0.0, 0.0],
            mass=1.0,
            node_id="A",
        )
    )

    simulation = Simulation(
        network,
        dt=0.1,
        update_materials=False,
        record_simulation_history=False,
    )

    recording = SimulationRecorder(
        simulation
    ).run_steps(
        4,
        sample_every=2,
        metadata={"name": "player test"},
    )

    path = tmp_path / "player_test.roifrec"

    RecordingStorage.save(
        recording,
        path,
    )

    return path


def test_parser_defaults():
    parser = build_parser()
    args = parser.parse_args(
        ["recording.roifrec"]
    )

    assert args.mode == "force"
    assert args.interval == 30.0
    assert args.frames is None
    assert args.repeat is False
    assert args.inspect is False


def test_inspect_mode_returns_success(
    tmp_path,
    capsys,
):
    path = make_recording_file(tmp_path)

    exit_code = main(
        [str(path), "--inspect"]
    )

    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Frames         : 3" in output
    assert "Physical steps : 4" in output
    assert "Inspection completed." in output


def test_missing_file_returns_code_two(
    tmp_path,
    capsys,
):
    missing = tmp_path / "missing.roifrec"

    exit_code = main(
        [str(missing), "--inspect"]
    )

    error = capsys.readouterr().err

    assert exit_code == 2
    assert "ERROR:" in error


def test_rejects_nonpositive_interval():
    parser = build_parser()
    args = parser.parse_args(
        [
            "recording.roifrec",
            "--interval",
            "0",
        ]
    )

    with pytest.raises(SystemExit):
        validate_arguments(parser, args)


def test_rejects_too_many_frames(
    tmp_path,
):
    path = make_recording_file(tmp_path)
    parser = build_parser()
    args = parser.parse_args(
        [
            str(path),
            "--frames",
            "100",
        ]
    )

    validate_arguments(parser, args)

    with pytest.raises(
        ValueError,
        match="cannot exceed recording length",
    ):
        run_player(args)


def test_gif_extension_validation():
    parser = build_parser()
    args = parser.parse_args(
        [
            "recording.roifrec",
            "--gif",
            "output.mp4",
        ]
    )

    with pytest.raises(SystemExit):
        validate_arguments(parser, args)
