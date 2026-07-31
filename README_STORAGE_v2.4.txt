Recording Storage v2.4
======================

Add or replace:

    core/storage.py
    core/__init__.py
    examples/example_19_recording_storage.py
    tests/test_storage.py

Run:

    python -m pytest
    python examples\example_19_recording_storage.py

Basic API:

    saved_path = RecordingStorage.save(
        recording,
        "output/run.roifrec",
        compress=True,
        overwrite=False,
    )

    info = RecordingStorage.inspect(
        saved_path,
        verify_checksum=True,
    )

    loaded = RecordingStorage.load(
        saved_path,
        verify_checksum=True,
    )

Important security note:

    Recording payloads use Python pickle because Network snapshots
    contain full Python object graphs. Only load .roifrec files from
    trusted sources.
