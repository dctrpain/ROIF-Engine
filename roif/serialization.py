"""
Versioned serialization utilities for ROIF Engine v1.

This module provides deterministic and safe persistence for ROIF objects and
JSON-compatible data. It supports JSON, optional YAML, CSV row collections,
compressed NumPy NPZ bundles, checksummed envelopes, and schema migrations.

Unsafe arbitrary-object formats such as pickle are intentionally excluded.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, TypeAlias, runtime_checkable
import csv
import hashlib
import io
import json
import math
import os
import tempfile

import numpy as np


class SerializationError(ValueError):
    """Raised when serialization input, format, or schema is invalid."""


JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
Decoder: TypeAlias = Callable[[Mapping[str, Any]], Any]
MigrationFunction: TypeAlias = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def _text(value: Any, *, name: str, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str):
        suffix = " or None" if optional else ""
        raise SerializationError(f"{name} must be a string{suffix}.")
    value = value.strip()
    if not value:
        raise SerializationError(f"{name} cannot be empty.")
    return value


def _path(value: str | os.PathLike[str]) -> Path:
    if isinstance(value, bool):
        raise SerializationError("path must be a filesystem path.")
    try:
        result = Path(value)
    except TypeError as exc:
        raise SerializationError("path must be a filesystem path.") from exc
    if not str(result).strip():
        raise SerializationError("path cannot be empty.")
    return result


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise SerializationError(f"{name} must be a real number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SerializationError(f"{name} must be a real number.") from exc
    if not math.isfinite(result):
        raise SerializationError(f"{name} must be finite.")
    return result


def _freeze(
    value: Mapping[str, Any] | None,
    *,
    name: str,
) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise SerializationError(f"{name} must be a mapping.")
    copied: dict[str, Any] = {}
    for key, item in value.items():
        copied[_text(key, name=f"{name} key")] = item
    return MappingProxyType(copied)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _timestamp(value: Any, *, name: str) -> str:
    text = _text(value, name=name)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SerializationError(f"{name} must be an ISO-8601 datetime.") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class SerializationFormat(str, Enum):
    JSON = "json"
    YAML = "yaml"
    CSV = "csv"
    NPZ = "npz"


class ChecksumAlgorithm(str, Enum):
    SHA256 = "sha256"
    SHA512 = "sha512"


@runtime_checkable
class SerializableObject(Protocol):
    def as_dict(self) -> Mapping[str, Any]:
        ...


def to_serializable(value: Any, *, allow_sets: bool = True) -> JSONValue:
    """Convert supported Python and ROIF objects to JSON-compatible data."""

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return _finite(value, name="float value")
    if isinstance(value, np.generic):
        return to_serializable(value.item(), allow_sets=allow_sets)
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number) and not np.all(np.isfinite(value)):
            raise SerializationError("NumPy arrays must contain finite values.")
        return to_serializable(value.tolist(), allow_sets=allow_sets)
    if isinstance(value, Enum):
        return to_serializable(value.value, allow_sets=allow_sets)
    if isinstance(value, SerializableObject):
        return to_serializable(value.as_dict(), allow_sets=allow_sets)
    if is_dataclass(value):
        return to_serializable(asdict(value), allow_sets=allow_sets)
    if isinstance(value, Mapping):
        return {
            _text(key, name="mapping key"): to_serializable(
                item, allow_sets=allow_sets
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [to_serializable(item, allow_sets=allow_sets) for item in value]
    if isinstance(value, (set, frozenset)):
        if not allow_sets:
            raise SerializationError("Set serialization is disabled.")
        items = [to_serializable(item, allow_sets=allow_sets) for item in value]
        return sorted(
            items,
            key=lambda item: json.dumps(
                item, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
        )
    if hasattr(value, "tolist") and callable(value.tolist):
        return to_serializable(value.tolist(), allow_sets=allow_sets)
    raise SerializationError(
        f"Unsupported serialization type: {type(value).__name__}."
    )


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        to_serializable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def compute_checksum(
    value: Any,
    *,
    algorithm: ChecksumAlgorithm = ChecksumAlgorithm.SHA256,
) -> str:
    if not isinstance(algorithm, ChecksumAlgorithm):
        raise SerializationError("algorithm must be a ChecksumAlgorithm.")
    hasher = hashlib.new(algorithm.value)
    hasher.update(canonical_json_bytes(value))
    return hasher.hexdigest()


@dataclass(frozen=True, slots=True)
class SerializationEnvelope:
    object_type: str
    schema_version: str
    created_at: str
    payload: JSONValue
    checksum: str
    checksum_algorithm: ChecksumAlgorithm = ChecksumAlgorithm.SHA256
    producer: str = "ROIF Engine"
    producer_version: str = "1.0.0"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "object_type", _text(self.object_type, name="object_type"))
        object.__setattr__(
            self, "schema_version", _text(self.schema_version, name="schema_version")
        )
        object.__setattr__(
            self, "created_at", _timestamp(self.created_at, name="created_at")
        )
        object.__setattr__(self, "payload", to_serializable(self.payload))

        checksum = _text(self.checksum, name="checksum")
        if not all(character in "0123456789abcdefABCDEF" for character in checksum):
            raise SerializationError("checksum must be hexadecimal.")
        object.__setattr__(self, "checksum", checksum.lower())

        if not isinstance(self.checksum_algorithm, ChecksumAlgorithm):
            raise SerializationError(
                "checksum_algorithm must be a ChecksumAlgorithm."
            )
        object.__setattr__(self, "producer", _text(self.producer, name="producer"))
        object.__setattr__(
            self,
            "producer_version",
            _text(self.producer_version, name="producer_version"),
        )
        object.__setattr__(
            self, "metadata", _freeze(self.metadata, name="metadata")
        )

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "object_type": self.object_type,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "payload": self.payload,
            "checksum": self.checksum,
            "checksum_algorithm": self.checksum_algorithm.value,
            "producer": self.producer,
            "producer_version": self.producer_version,
            "metadata": to_serializable(self.metadata),
        }


def make_envelope(
    payload: Any,
    *,
    object_type: str,
    schema_version: str = "1.0",
    producer: str = "ROIF Engine",
    producer_version: str = "1.0.0",
    checksum_algorithm: ChecksumAlgorithm = ChecksumAlgorithm.SHA256,
    created_at: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SerializationEnvelope:
    serialized = to_serializable(payload)
    return SerializationEnvelope(
        object_type=object_type,
        schema_version=schema_version,
        created_at=utc_now_iso() if created_at is None else created_at,
        payload=serialized,
        checksum=compute_checksum(serialized, algorithm=checksum_algorithm),
        checksum_algorithm=checksum_algorithm,
        producer=producer,
        producer_version=producer_version,
        metadata={} if metadata is None else metadata,
    )


def envelope_from_mapping(value: Mapping[str, Any]) -> SerializationEnvelope:
    if not isinstance(value, Mapping):
        raise SerializationError("Envelope value must be a mapping.")

    required = {
        "object_type",
        "schema_version",
        "created_at",
        "payload",
        "checksum",
        "checksum_algorithm",
        "producer",
        "producer_version",
        "metadata",
    }
    missing = required - set(value)
    if missing:
        raise SerializationError(
            "Envelope is missing required fields: "
            + ", ".join(sorted(missing))
            + "."
        )
    unknown = set(value) - required
    if unknown:
        raise SerializationError(
            "Envelope contains unknown fields: "
            + ", ".join(sorted(str(item) for item in unknown))
            + "."
        )
    try:
        algorithm = ChecksumAlgorithm(value["checksum_algorithm"])
    except (TypeError, ValueError) as exc:
        raise SerializationError("Invalid checksum_algorithm.") from exc

    return SerializationEnvelope(
        object_type=value["object_type"],
        schema_version=value["schema_version"],
        created_at=value["created_at"],
        payload=value["payload"],
        checksum=value["checksum"],
        checksum_algorithm=algorithm,
        producer=value["producer"],
        producer_version=value["producer_version"],
        metadata=value["metadata"],
    )


def verify_envelope(envelope: SerializationEnvelope) -> bool:
    if not isinstance(envelope, SerializationEnvelope):
        raise SerializationError("envelope must be a SerializationEnvelope.")
    expected = compute_checksum(
        envelope.payload,
        algorithm=envelope.checksum_algorithm,
    )
    return expected == envelope.checksum


def require_valid_envelope(envelope: SerializationEnvelope) -> None:
    if not verify_envelope(envelope):
        raise SerializationError("Envelope checksum verification failed.")


@dataclass(frozen=True, slots=True)
class MigrationStep:
    object_type: str
    from_version: str
    to_version: str
    migrate: MigrationFunction

    def __post_init__(self) -> None:
        object.__setattr__(self, "object_type", _text(self.object_type, name="object_type"))
        object.__setattr__(
            self, "from_version", _text(self.from_version, name="from_version")
        )
        object.__setattr__(self, "to_version", _text(self.to_version, name="to_version"))
        if self.from_version == self.to_version:
            raise SerializationError("Migration versions must differ.")
        if not callable(self.migrate):
            raise SerializationError("migrate must be callable.")


class MigrationRegistry:
    """Registry of deterministic forward migrations."""

    def __init__(self) -> None:
        self._steps: dict[tuple[str, str], MigrationStep] = {}

    def register(self, step: MigrationStep) -> "MigrationRegistry":
        if not isinstance(step, MigrationStep):
            raise SerializationError("step must be a MigrationStep.")
        key = (step.object_type, step.from_version)
        if key in self._steps:
            raise SerializationError(
                f"A migration is already registered for {step.object_type} "
                f"{step.from_version}."
            )
        self._steps[key] = step
        return self

    def unregister(
        self,
        object_type: str,
        from_version: str,
    ) -> "MigrationRegistry":
        self._steps.pop(
            (
                _text(object_type, name="object_type"),
                _text(from_version, name="from_version"),
            ),
            None,
        )
        return self

    def migrate_payload(
        self,
        *,
        object_type: str,
        payload: Mapping[str, Any],
        from_version: str,
        to_version: str,
        maximum_steps: int = 100,
    ) -> Mapping[str, Any]:
        object_type = _text(object_type, name="object_type")
        current_version = _text(from_version, name="from_version")
        target_version = _text(to_version, name="to_version")
        if not isinstance(payload, Mapping):
            raise SerializationError("payload must be a mapping.")
        if (
            isinstance(maximum_steps, bool)
            or not isinstance(maximum_steps, int)
            or maximum_steps < 1
        ):
            raise SerializationError("maximum_steps must be at least 1.")

        current: Mapping[str, Any] = dict(payload)
        visited: set[str] = set()
        step_count = 0

        while current_version != target_version:
            if current_version in visited:
                raise SerializationError("Migration cycle detected.")
            visited.add(current_version)

            step = self._steps.get((object_type, current_version))
            if step is None:
                raise SerializationError(
                    f"No migration path from {object_type} {current_version} "
                    f"to {target_version}."
                )

            migrated = step.migrate(dict(current))
            if not isinstance(migrated, Mapping):
                raise SerializationError(
                    "Migration functions must return mappings."
                )
            current = dict(migrated)
            current_version = step.to_version
            step_count += 1
            if step_count > maximum_steps:
                raise SerializationError("Migration exceeded maximum_steps.")

        return MappingProxyType(dict(current))

    def migrate_envelope(
        self,
        envelope: SerializationEnvelope,
        *,
        to_version: str,
    ) -> SerializationEnvelope:
        require_valid_envelope(envelope)
        if not isinstance(envelope.payload, Mapping):
            raise SerializationError("Only mapping payloads can be migrated.")
        migrated = self.migrate_payload(
            object_type=envelope.object_type,
            payload=envelope.payload,
            from_version=envelope.schema_version,
            to_version=to_version,
        )
        return make_envelope(
            migrated,
            object_type=envelope.object_type,
            schema_version=to_version,
            producer=envelope.producer,
            producer_version=envelope.producer_version,
            checksum_algorithm=envelope.checksum_algorithm,
            metadata={
                **dict(envelope.metadata),
                "migrated_from": envelope.schema_version,
            },
        )


def dumps_json(
    value: Any,
    *,
    indent: int | None = 2,
    sort_keys: bool = False,
) -> str:
    if indent is not None and (
        isinstance(indent, bool)
        or not isinstance(indent, int)
        or indent < 0
    ):
        raise SerializationError(
            "indent must be a non-negative integer or None."
        )
    if not isinstance(sort_keys, bool):
        raise SerializationError("sort_keys must be a bool.")
    return json.dumps(
        to_serializable(value),
        ensure_ascii=False,
        indent=indent,
        sort_keys=sort_keys,
        allow_nan=False,
    )


def loads_json(text: str) -> JSONValue:
    text = _text(text, name="text")
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SerializationError(f"Invalid JSON: {exc.msg}.") from exc
    return to_serializable(value)


def _atomic_write(path: Path, text: str, *, encoding: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding=encoding,
        newline="",
        delete=False,
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(handle.name)
    try:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(temporary, path)
    except Exception:
        handle.close()
        if temporary.exists():
            temporary.unlink()
        raise


def save_json(
    value: Any,
    path: str | os.PathLike[str],
    *,
    indent: int | None = 2,
    sort_keys: bool = False,
    encoding: str = "utf-8",
    atomic: bool = True,
) -> Path:
    path = _path(path)
    encoding = _text(encoding, name="encoding")
    if not isinstance(atomic, bool):
        raise SerializationError("atomic must be a bool.")
    text = dumps_json(value, indent=indent, sort_keys=sort_keys)
    if atomic:
        _atomic_write(path, text, encoding=encoding)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding=encoding)
    return path


def load_json(
    path: str | os.PathLike[str],
    *,
    encoding: str = "utf-8",
) -> JSONValue:
    path = _path(path)
    encoding = _text(encoding, name="encoding")
    if not path.is_file():
        raise SerializationError(f"JSON file does not exist: {path}.")
    try:
        return loads_json(path.read_text(encoding=encoding))
    except OSError as exc:
        raise SerializationError(f"Unable to read JSON file: {path}.") from exc


def load_json_envelope(
    path: str | os.PathLike[str],
    *,
    verify: bool = True,
    encoding: str = "utf-8",
) -> SerializationEnvelope:
    if not isinstance(verify, bool):
        raise SerializationError("verify must be a bool.")
    value = load_json(path, encoding=encoding)
    if not isinstance(value, Mapping):
        raise SerializationError("Envelope JSON must contain an object.")
    envelope = envelope_from_mapping(value)
    if verify:
        require_valid_envelope(envelope)
    return envelope


def decode_envelope(
    envelope: SerializationEnvelope,
    decoder: Decoder,
    *,
    verify: bool = True,
) -> Any:
    if not isinstance(envelope, SerializationEnvelope):
        raise SerializationError("envelope must be a SerializationEnvelope.")
    if not callable(decoder):
        raise SerializationError("decoder must be callable.")
    if not isinstance(verify, bool):
        raise SerializationError("verify must be a bool.")
    if verify:
        require_valid_envelope(envelope)
    if not isinstance(envelope.payload, Mapping):
        raise SerializationError("Decoder requires a mapping payload.")
    return decoder(envelope.payload)


def dumps_yaml(value: Any) -> str:
    try:
        import yaml
    except ImportError as exc:
        raise SerializationError(
            "PyYAML is required for YAML serialization."
        ) from exc
    return yaml.safe_dump(
        to_serializable(value),
        allow_unicode=True,
        sort_keys=False,
    )


def loads_yaml(text: str) -> JSONValue:
    text = _text(text, name="text")
    try:
        import yaml
    except ImportError as exc:
        raise SerializationError(
            "PyYAML is required for YAML serialization."
        ) from exc
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SerializationError("Invalid YAML input.") from exc
    return to_serializable(value)


def save_yaml(
    value: Any,
    path: str | os.PathLike[str],
    *,
    encoding: str = "utf-8",
    atomic: bool = True,
) -> Path:
    path = _path(path)
    encoding = _text(encoding, name="encoding")
    if not isinstance(atomic, bool):
        raise SerializationError("atomic must be a bool.")
    text = dumps_yaml(value)
    if atomic:
        _atomic_write(path, text, encoding=encoding)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding=encoding)
    return path


def load_yaml(
    path: str | os.PathLike[str],
    *,
    encoding: str = "utf-8",
) -> JSONValue:
    path = _path(path)
    encoding = _text(encoding, name="encoding")
    if not path.is_file():
        raise SerializationError(f"YAML file does not exist: {path}.")
    try:
        return loads_yaml(path.read_text(encoding=encoding))
    except OSError as exc:
        raise SerializationError(f"Unable to read YAML file: {path}.") from exc


def _csv_rows(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, JSONScalar], ...]:
    if isinstance(rows, (str, bytes)):
        raise SerializationError("rows must be an iterable of mappings.")
    try:
        values = tuple(rows)
    except TypeError as exc:
        raise SerializationError(
            "rows must be an iterable of mappings."
        ) from exc

    parsed: list[dict[str, JSONScalar]] = []
    for index, row in enumerate(values):
        if not isinstance(row, Mapping):
            raise SerializationError(f"CSV row {index} must be a mapping.")
        output: dict[str, JSONScalar] = {}
        for key, value in row.items():
            column = _text(key, name=f"CSV row {index} column")
            serialized = to_serializable(value)
            if isinstance(serialized, (list, dict)):
                serialized = json.dumps(
                    serialized,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            output[column] = serialized
        parsed.append(output)
    return tuple(parsed)


def dumps_csv(
    rows: Iterable[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
    delimiter: str = ",",
    lineterminator: str = "\n",
) -> str:
    rows = _csv_rows(rows)
    delimiter = _text(delimiter, name="delimiter")
    if len(delimiter) != 1:
        raise SerializationError(
            "delimiter must contain exactly one character."
        )
    if not isinstance(lineterminator, str):
        raise SerializationError("lineterminator must be a string.")
    if lineterminator == "":
        raise SerializationError("lineterminator cannot be empty.")

    if fieldnames is None:
        ordered: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    ordered.append(key)
        columns = tuple(ordered)
    else:
        if isinstance(fieldnames, (str, bytes)):
            raise SerializationError(
                "fieldnames must be a sequence of strings."
            )
        columns = tuple(_text(item, name="fieldname") for item in fieldnames)
        if len(columns) != len(set(columns)):
            raise SerializationError(
                "fieldnames cannot contain duplicates."
            )

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=columns,
        delimiter=delimiter,
        lineterminator=lineterminator,
        extrasaction="raise",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def loads_csv(
    text: str,
    *,
    delimiter: str = ",",
) -> tuple[Mapping[str, str], ...]:
    text = _text(text, name="text")
    delimiter = _text(delimiter, name="delimiter")
    if len(delimiter) != 1:
        raise SerializationError(
            "delimiter must contain exactly one character."
        )

    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    if reader.fieldnames is None:
        raise SerializationError("CSV input must contain a header row.")

    fieldnames = [_text(item, name="CSV fieldname") for item in reader.fieldnames]
    if len(fieldnames) != len(set(fieldnames)):
        raise SerializationError(
            "CSV header cannot contain duplicate fieldnames."
        )

    result: list[Mapping[str, str]] = []
    for row in reader:
        if None in row:
            raise SerializationError(
                "CSV row contains extra unnamed values."
            )
        result.append(
            MappingProxyType({
                key: "" if value is None else value
                for key, value in row.items()
            })
        )
    return tuple(result)


def save_csv(
    rows: Iterable[Mapping[str, Any]],
    path: str | os.PathLike[str],
    *,
    fieldnames: Sequence[str] | None = None,
    delimiter: str = ",",
    encoding: str = "utf-8",
    atomic: bool = True,
) -> Path:
    path = _path(path)
    encoding = _text(encoding, name="encoding")
    if not isinstance(atomic, bool):
        raise SerializationError("atomic must be a bool.")
    text = dumps_csv(rows, fieldnames=fieldnames, delimiter=delimiter)
    if atomic:
        _atomic_write(path, text, encoding=encoding)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding=encoding)
    return path


def load_csv(
    path: str | os.PathLike[str],
    *,
    delimiter: str = ",",
    encoding: str = "utf-8",
) -> tuple[Mapping[str, str], ...]:
    path = _path(path)
    encoding = _text(encoding, name="encoding")
    if not path.is_file():
        raise SerializationError(f"CSV file does not exist: {path}.")
    try:
        return loads_csv(
            path.read_text(encoding=encoding),
            delimiter=delimiter,
        )
    except OSError as exc:
        raise SerializationError(f"Unable to read CSV file: {path}.") from exc


@dataclass(frozen=True, slots=True)
class NPZBundle:
    arrays: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.arrays, Mapping):
            raise SerializationError("arrays must be a mapping.")
        parsed: dict[str, np.ndarray] = {}
        for key, value in self.arrays.items():
            name = _text(key, name="array name")
            try:
                array = np.asarray(value)
            except (TypeError, ValueError) as exc:
                raise SerializationError(
                    f"Array {name!r} could not be converted."
                ) from exc
            if array.dtype == object:
                raise SerializationError(
                    f"Array {name!r} cannot use object dtype."
                )
            if np.issubdtype(array.dtype, np.number) and not np.all(
                np.isfinite(array)
            ):
                raise SerializationError(
                    f"Array {name!r} must contain finite values."
                )
            copied = np.array(array, copy=True)
            copied.setflags(write=False)
            parsed[name] = copied
        object.__setattr__(self, "arrays", MappingProxyType(parsed))
        object.__setattr__(
            self, "metadata", _freeze(self.metadata, name="metadata")
        )

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "arrays": {
                key: value.tolist()
                for key, value in self.arrays.items()
            },
            "metadata": to_serializable(self.metadata),
        }


def save_npz(
    arrays: Mapping[str, Any] | NPZBundle,
    path: str | os.PathLike[str],
    *,
    metadata: Mapping[str, Any] | None = None,
    compressed: bool = True,
) -> Path:
    path = _path(path)
    if not isinstance(compressed, bool):
        raise SerializationError("compressed must be a bool.")

    if isinstance(arrays, NPZBundle):
        if metadata is not None:
            raise SerializationError(
                "metadata cannot be supplied with an NPZBundle."
            )
        bundle = arrays
    else:
        bundle = NPZBundle(
            arrays=arrays,
            metadata={} if metadata is None else metadata,
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(bundle.arrays)
    payload["__metadata_json__"] = np.asarray(
        dumps_json(bundle.metadata, indent=None, sort_keys=True),
        dtype=np.str_,
    )
    try:
        if compressed:
            np.savez_compressed(path, **payload)
        else:
            np.savez(path, **payload)
    except OSError as exc:
        raise SerializationError(f"Unable to save NPZ file: {path}.") from exc
    return path


def load_npz(path: str | os.PathLike[str]) -> NPZBundle:
    path = _path(path)
    if not path.is_file():
        raise SerializationError(f"NPZ file does not exist: {path}.")
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays: dict[str, np.ndarray] = {}
            metadata_text = "{}"
            for key in archive.files:
                if key == "__metadata_json__":
                    metadata_text = str(archive[key].item())
                else:
                    arrays[key] = np.array(archive[key], copy=True)
    except (OSError, ValueError) as exc:
        raise SerializationError(f"Unable to load NPZ file: {path}.") from exc

    metadata = loads_json(metadata_text)
    if not isinstance(metadata, Mapping):
        raise SerializationError("NPZ metadata must be a mapping.")
    return NPZBundle(arrays=arrays, metadata=metadata)


def infer_format(path: str | os.PathLike[str]) -> SerializationFormat:
    suffix = _path(path).suffix.lower()
    formats = {
        ".json": SerializationFormat.JSON,
        ".yaml": SerializationFormat.YAML,
        ".yml": SerializationFormat.YAML,
        ".csv": SerializationFormat.CSV,
        ".npz": SerializationFormat.NPZ,
    }
    try:
        return formats[suffix]
    except KeyError as exc:
        raise SerializationError(
            f"Unsupported file extension: {suffix or '<none>'}."
        ) from exc


__all__ = [
    "ChecksumAlgorithm",
    "Decoder",
    "JSONScalar",
    "JSONValue",
    "MigrationFunction",
    "MigrationRegistry",
    "MigrationStep",
    "NPZBundle",
    "SerializableObject",
    "SerializationEnvelope",
    "SerializationError",
    "SerializationFormat",
    "canonical_json_bytes",
    "compute_checksum",
    "decode_envelope",
    "dumps_csv",
    "dumps_json",
    "dumps_yaml",
    "envelope_from_mapping",
    "infer_format",
    "load_csv",
    "load_json",
    "load_json_envelope",
    "load_npz",
    "load_yaml",
    "loads_csv",
    "loads_json",
    "loads_yaml",
    "make_envelope",
    "require_valid_envelope",
    "save_csv",
    "save_json",
    "save_npz",
    "save_yaml",
    "to_serializable",
    "utc_now_iso",
    "verify_envelope",
]

