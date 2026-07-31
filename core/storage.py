from __future__ import annotations

import gzip
import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from .recorder import (
    RecordingFrame,
    SimulationRecording,
)


class RecordingStorageError(RuntimeError):
    """Base error raised by Recording Storage."""


class RecordingFormatError(RecordingStorageError):
    """Raised when a file is not a supported recording."""


class RecordingIntegrityError(RecordingStorageError):
    """Raised when checksum or payload validation fails."""


@dataclass(frozen=True, slots=True)
class RecordingFileInfo:
    """
    Metadata returned by inspect() without loading frame payloads.
    """

    path: Path
    format_name: str
    format_version: int
    storage_version: str
    compressed: bool
    frame_count: int
    simulation_dt: float
    sample_every: int
    start_time: float
    end_time: float
    duration: float
    physical_steps: int
    metadata: dict[str, Any]
    payload_sha256: str
    file_size_bytes: int


class RecordingStorage:
    """
    Persistent storage for SimulationRecording.

    File format:
        .roifrec       gzip-compressed container
        .roifrec.gz    gzip-compressed container
        .roifrec.raw   uncompressed container

    The container stores:
        - a JSON-compatible manifest;
        - a pickle payload containing SimulationRecording;
        - SHA-256 checksum of the payload.

    Pickle is appropriate here because Network snapshots contain Python
    objects that cannot be represented faithfully by plain JSON/NPZ.

    Security:
        Loading pickle data from untrusted files is unsafe. Only load
        recordings created by your own ROIF Engine installation or a
        trusted collaborator.
    """

    VERSION = "2.4"
    FORMAT_NAME = "ROIF Simulation Recording"
    FORMAT_VERSION = 1
    MAGIC = b"ROIFREC\x00"
    HEADER_SIZE_BYTES = 8

    @classmethod
    def save(
        cls,
        recording: SimulationRecording,
        path: str | Path,
        *,
        compress: bool = True,
        overwrite: bool = False,
        protocol: int = pickle.HIGHEST_PROTOCOL,
    ) -> Path:
        """
        Save a complete SimulationRecording to disk.

        Returns the resolved output path.
        """

        if not isinstance(
            recording,
            SimulationRecording,
        ):
            raise TypeError(
                "recording must be a SimulationRecording"
            )

        output_path = Path(path)

        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"recording already exists: {output_path}"
            )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = pickle.dumps(
            recording,
            protocol=int(protocol),
        )
        payload_sha256 = hashlib.sha256(
            payload
        ).hexdigest()

        manifest = cls._build_manifest(
            recording=recording,
            compressed=bool(compress),
            payload_sha256=payload_sha256,
            payload_size_bytes=len(payload),
        )
        manifest_bytes = json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

        temporary_path = output_path.with_name(
            output_path.name + ".tmp"
        )

        opener = (
            gzip.open
            if compress
            else open
        )

        try:
            with opener(
                temporary_path,
                "wb",
            ) as stream:
                cls._write_container(
                    stream,
                    manifest_bytes,
                    payload,
                )

            temporary_path.replace(output_path)
        except Exception:
            temporary_path.unlink(
                missing_ok=True
            )
            raise

        return output_path.resolve()

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        verify_checksum: bool = True,
    ) -> SimulationRecording:
        """
        Load a SimulationRecording from disk.

        Only load trusted files because the payload uses pickle.
        """

        input_path = cls._validate_input_path(path)
        manifest, payload = cls._read_container_auto(
            input_path
        )

        cls._validate_manifest(manifest)

        if verify_checksum:
            cls._verify_payload_checksum(
                manifest,
                payload,
            )

        try:
            recording = pickle.loads(payload)
        except Exception as exc:
            raise RecordingFormatError(
                "recording payload could not be decoded"
            ) from exc

        if not isinstance(
            recording,
            SimulationRecording,
        ):
            raise RecordingFormatError(
                "payload is not a SimulationRecording"
            )

        cls._validate_recording_against_manifest(
            recording,
            manifest,
        )

        return recording

    @classmethod
    def inspect(
        cls,
        path: str | Path,
        *,
        verify_checksum: bool = False,
    ) -> RecordingFileInfo:
        """
        Read file metadata.

        inspect() does not unpickle the recording, but it reads the
        payload bytes so checksum verification can be requested.
        """

        input_path = cls._validate_input_path(path)
        manifest, payload = cls._read_container_auto(
            input_path
        )

        cls._validate_manifest(manifest)

        if verify_checksum:
            cls._verify_payload_checksum(
                manifest,
                payload,
            )

        recording_info = manifest["recording"]

        return RecordingFileInfo(
            path=input_path.resolve(),
            format_name=str(
                manifest["format_name"]
            ),
            format_version=int(
                manifest["format_version"]
            ),
            storage_version=str(
                manifest["storage_version"]
            ),
            compressed=bool(
                manifest["compressed"]
            ),
            frame_count=int(
                recording_info["frame_count"]
            ),
            simulation_dt=float(
                recording_info["simulation_dt"]
            ),
            sample_every=int(
                recording_info["sample_every"]
            ),
            start_time=float(
                recording_info["start_time"]
            ),
            end_time=float(
                recording_info["end_time"]
            ),
            duration=float(
                recording_info["duration"]
            ),
            physical_steps=int(
                recording_info["physical_steps"]
            ),
            metadata=dict(
                recording_info.get(
                    "metadata",
                    {},
                )
            ),
            payload_sha256=str(
                manifest["payload"]["sha256"]
            ),
            file_size_bytes=input_path.stat().st_size,
        )

    @classmethod
    def _build_manifest(
        cls,
        *,
        recording: SimulationRecording,
        compressed: bool,
        payload_sha256: str,
        payload_size_bytes: int,
    ) -> dict[str, Any]:
        return {
            "format_name": cls.FORMAT_NAME,
            "format_version": cls.FORMAT_VERSION,
            "storage_version": cls.VERSION,
            "compressed": compressed,
            "recording": {
                "recording_version": (
                    recording.VERSION
                ),
                "frame_count": len(recording),
                "simulation_dt": (
                    recording.simulation_dt
                ),
                "sample_every": (
                    recording.sample_every
                ),
                "start_time": (
                    recording.start_time
                ),
                "end_time": recording.end_time,
                "duration": recording.duration,
                "physical_steps": (
                    recording.physical_steps
                ),
                "metadata": cls._json_safe(
                    recording.metadata
                ),
            },
            "payload": {
                "encoding": "python-pickle",
                "sha256": payload_sha256,
                "size_bytes": payload_size_bytes,
            },
        }

    @classmethod
    def _write_container(
        cls,
        stream: BinaryIO,
        manifest_bytes: bytes,
        payload: bytes,
    ) -> None:
        stream.write(cls.MAGIC)
        stream.write(
            len(manifest_bytes).to_bytes(
                cls.HEADER_SIZE_BYTES,
                byteorder="big",
                signed=False,
            )
        )
        stream.write(manifest_bytes)
        stream.write(payload)

    @classmethod
    def _read_container_auto(
        cls,
        path: Path,
    ) -> tuple[dict[str, Any], bytes]:
        raw_prefix = path.read_bytes()[:2]
        is_gzip = raw_prefix == b"\x1f\x8b"

        opener = gzip.open if is_gzip else open

        try:
            with opener(path, "rb") as stream:
                return cls._read_container(
                    stream
                )
        except RecordingStorageError:
            raise
        except Exception as exc:
            raise RecordingFormatError(
                "could not read recording container"
            ) from exc

    @classmethod
    def _read_container(
        cls,
        stream: BinaryIO,
    ) -> tuple[dict[str, Any], bytes]:
        magic = stream.read(len(cls.MAGIC))

        if magic != cls.MAGIC:
            raise RecordingFormatError(
                "invalid ROIF recording signature"
            )

        header_size_raw = stream.read(
            cls.HEADER_SIZE_BYTES
        )

        if len(header_size_raw) != cls.HEADER_SIZE_BYTES:
            raise RecordingFormatError(
                "recording header is truncated"
            )

        manifest_size = int.from_bytes(
            header_size_raw,
            byteorder="big",
            signed=False,
        )

        if manifest_size <= 0:
            raise RecordingFormatError(
                "recording manifest size is invalid"
            )

        manifest_bytes = stream.read(
            manifest_size
        )

        if len(manifest_bytes) != manifest_size:
            raise RecordingFormatError(
                "recording manifest is truncated"
            )

        try:
            manifest = json.loads(
                manifest_bytes.decode("utf-8")
            )
        except Exception as exc:
            raise RecordingFormatError(
                "recording manifest is invalid"
            ) from exc

        payload = stream.read()

        if not payload:
            raise RecordingFormatError(
                "recording payload is empty"
            )

        return manifest, payload

    @classmethod
    def _validate_manifest(
        cls,
        manifest: dict[str, Any],
    ) -> None:
        if not isinstance(manifest, dict):
            raise RecordingFormatError(
                "recording manifest must be an object"
            )

        if (
            manifest.get("format_name")
            != cls.FORMAT_NAME
        ):
            raise RecordingFormatError(
                "unsupported recording format"
            )

        if (
            int(
                manifest.get(
                    "format_version",
                    -1,
                )
            )
            != cls.FORMAT_VERSION
        ):
            raise RecordingFormatError(
                "unsupported recording format version"
            )

        recording = manifest.get("recording")
        payload = manifest.get("payload")

        if not isinstance(recording, dict):
            raise RecordingFormatError(
                "recording metadata is missing"
            )

        if not isinstance(payload, dict):
            raise RecordingFormatError(
                "payload metadata is missing"
            )

        required_recording_keys = {
            "frame_count",
            "simulation_dt",
            "sample_every",
            "start_time",
            "end_time",
            "duration",
            "physical_steps",
        }

        if not required_recording_keys.issubset(
            recording
        ):
            raise RecordingFormatError(
                "recording metadata is incomplete"
            )

        if payload.get("encoding") != "python-pickle":
            raise RecordingFormatError(
                "unsupported recording payload encoding"
            )

        checksum = payload.get("sha256")
        if (
            not isinstance(checksum, str)
            or len(checksum) != 64
        ):
            raise RecordingFormatError(
                "recording checksum is invalid"
            )

    @classmethod
    def _verify_payload_checksum(
        cls,
        manifest: dict[str, Any],
        payload: bytes,
    ) -> None:
        expected = manifest["payload"]["sha256"]
        actual = hashlib.sha256(
            payload
        ).hexdigest()

        if actual != expected:
            raise RecordingIntegrityError(
                "recording payload checksum mismatch"
            )

        expected_size = int(
            manifest["payload"]["size_bytes"]
        )

        if len(payload) != expected_size:
            raise RecordingIntegrityError(
                "recording payload size mismatch"
            )

    @classmethod
    def _validate_recording_against_manifest(
        cls,
        recording: SimulationRecording,
        manifest: dict[str, Any],
    ) -> None:
        info = manifest["recording"]

        checks = {
            "frame_count": (
                len(recording),
                int(info["frame_count"]),
            ),
            "sample_every": (
                recording.sample_every,
                int(info["sample_every"]),
            ),
            "physical_steps": (
                recording.physical_steps,
                int(info["physical_steps"]),
            ),
        }

        for name, (actual, expected) in checks.items():
            if actual != expected:
                raise RecordingIntegrityError(
                    f"recording {name} does not match manifest"
                )

        float_checks = {
            "simulation_dt": (
                recording.simulation_dt,
                float(info["simulation_dt"]),
            ),
            "start_time": (
                recording.start_time,
                float(info["start_time"]),
            ),
            "end_time": (
                recording.end_time,
                float(info["end_time"]),
            ),
            "duration": (
                recording.duration,
                float(info["duration"]),
            ),
        }

        for name, (actual, expected) in float_checks.items():
            if abs(actual - expected) > 1e-12:
                raise RecordingIntegrityError(
                    f"recording {name} does not match manifest"
                )

    @staticmethod
    def _validate_input_path(
        path: str | Path,
    ) -> Path:
        input_path = Path(path)

        if not input_path.exists():
            raise FileNotFoundError(
                f"recording does not exist: {input_path}"
            )

        if not input_path.is_file():
            raise IsADirectoryError(
                f"recording path is not a file: {input_path}"
            )

        return input_path

    @classmethod
    def _json_safe(
        cls,
        value: Any,
    ) -> Any:
        """
        Convert metadata to a JSON-safe representation.

        The original metadata remains preserved inside the pickle
        payload. The manifest representation is intended for inspection.
        """

        if value is None:
            return None

        if isinstance(
            value,
            (str, int, float, bool),
        ):
            return value

        if isinstance(value, dict):
            return {
                str(key): cls._json_safe(item)
                for key, item in value.items()
            }

        if isinstance(
            value,
            (list, tuple, set),
        ):
            return [
                cls._json_safe(item)
                for item in value
            ]

        return repr(value)
