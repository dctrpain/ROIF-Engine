from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from core.network import Network
from core.node import Node
from core.recorder import SimulationRecorder
from core.simulation import Simulation
from core.storage import (
    RecordingFormatError,
    RecordingIntegrityError,
    RecordingStorage,
)


def make_recording():
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
        metadata={
            "name": "storage test",
        },
    )

    return simulation, recording


def test_save_and_load_compressed(tmp_path):
    _, recording = make_recording()
    path = tmp_path / "test.roifrec"

    saved = RecordingStorage.save(
        recording,
        path,
    )
    loaded = RecordingStorage.load(saved)

    assert saved.exists()
    assert len(loaded) == len(recording)
    assert (
        loaded.last_frame.step_index
        == recording.last_frame.step_index
    )
    assert loaded.end_time == pytest.approx(
        recording.end_time
    )
    assert loaded.metadata == recording.metadata


def test_save_and_load_uncompressed(tmp_path):
    _, recording = make_recording()
    path = tmp_path / "test.roifrec.raw"

    RecordingStorage.save(
        recording,
        path,
        compress=False,
    )
    loaded = RecordingStorage.load(path)

    assert len(loaded) == len(recording)


def test_inspect_without_unpickling(tmp_path):
    _, recording = make_recording()
    path = tmp_path / "inspect.roifrec"

    RecordingStorage.save(
        recording,
        path,
    )

    info = RecordingStorage.inspect(
        path,
        verify_checksum=True,
    )

    assert info.format_version == 1
    assert info.storage_version == "2.4"
    assert info.compressed is True
    assert info.frame_count == 3
    assert info.physical_steps == 4
    assert info.duration == pytest.approx(0.4)
    assert info.metadata["name"] == "storage test"
    assert len(info.payload_sha256) == 64
    assert info.file_size_bytes > 0


def test_save_does_not_overwrite_by_default(tmp_path):
    _, recording = make_recording()
    path = tmp_path / "existing.roifrec"

    RecordingStorage.save(
        recording,
        path,
    )

    with pytest.raises(FileExistsError):
        RecordingStorage.save(
            recording,
            path,
        )


def test_save_can_overwrite(tmp_path):
    _, recording = make_recording()
    path = tmp_path / "overwrite.roifrec"

    RecordingStorage.save(
        recording,
        path,
    )
    RecordingStorage.save(
        recording,
        path,
        overwrite=True,
    )

    assert path.exists()


def test_invalid_signature_is_rejected(tmp_path):
    path = tmp_path / "invalid.roifrec"
    path.write_bytes(b"not a recording")

    with pytest.raises(
        RecordingFormatError,
        match="signature",
    ):
        RecordingStorage.load(path)


def test_corrupted_payload_is_detected(tmp_path):
    _, recording = make_recording()
    path = tmp_path / "corrupt.roifrec"

    RecordingStorage.save(
        recording,
        path,
        compress=False,
    )

    data = bytearray(path.read_bytes())
    data[-1] ^= 0xFF
    path.write_bytes(data)

    with pytest.raises(
        RecordingIntegrityError,
        match="checksum",
    ):
        RecordingStorage.load(
            path,
            verify_checksum=True,
        )


def test_compressed_file_has_gzip_signature(tmp_path):
    _, recording = make_recording()
    path = tmp_path / "compressed.roifrec"

    RecordingStorage.save(
        recording,
        path,
        compress=True,
    )

    assert path.read_bytes()[:2] == b"\x1f\x8b"

    with gzip.open(path, "rb") as stream:
        assert stream.read(
            len(RecordingStorage.MAGIC)
        ) == RecordingStorage.MAGIC


def test_storage_does_not_mutate_simulation(tmp_path):
    simulation, recording = make_recording()
    path = tmp_path / "stable.roifrec"

    step_before = simulation.step_index
    time_before = simulation.time

    RecordingStorage.save(
        recording,
        path,
    )
    _ = RecordingStorage.inspect(path)
    _ = RecordingStorage.load(path)

    assert simulation.step_index == step_before
    assert simulation.time == time_before
